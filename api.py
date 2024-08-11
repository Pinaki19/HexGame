import asyncio
import json
import random
import requests
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi import Query
from fastapi.websockets import WebSocketDisconnect
import uuid
import os
import uvicorn
# Import pymongo
import pymongo
from datetime import datetime
from typing import Union, Dict, List, Set, Optional
import onnxruntime
import numpy as np
from datetime import datetime

underway=False
session = None

class Move(BaseModel):
    row:int
    column:int
    blue:bool
    
class Disc_data(BaseModel):
    id:str
    name:str

class User(BaseModel):
    name: str
    profile_pic: str
    
class gameRequest(BaseModel):
    id:str
    

class InputData(BaseModel):
    # Define the input data model using Pydantic
    matrix: list
    player: str
    agent: str
    agent_is_blue: bool

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = None

game_mapping: Dict[str, List[str]] = {}                 #stores a game_id vs list of 2 session_ids
cur_games: Dict[str,str] = {}                           #current ongoing games
connections: Dict[str, Dict[str,WebSocket]] = {}        #stores a web_socket connection against a game and session
game_coroutines = {}                                    #helps start game in async mode
turn_counts={}                                          # how many turns have benn played in a game
turn_details={}                                         #stores who had the first turn
start_times={}                                          #when did a game start?
watch_list={}                                           # list of sockets watching this game
game_participants={}                                    #names of participants for a game
socket_list={}                                          #sockets for a game, ind 0 is blue's socket

def handle_click(matrix, player, agent, agent_is_blue):
    def PosToId(x, y):
        return x + 11 * y

    def has_winner(board):
        # Define directions for hexagon grid
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        # Function to check if a cell is valid
        def is_valid_cell(row, col, check):
            return row >= 0 and row < 11 and col >= 0 and col < 11 and board[col * 11 + row] == check

        # Function to perform DFS traversal
        def dfs(row, col, visited, target_row, target_col, check):
            visited.add(col * 11 + row)
            # If we reach the opposite side, we have a winner
            if (target_row == 10 and row == target_row) or (target_col == 10 and col == target_col):
                return True

            # Check all neighbors
            for dx, dy in directions:
                new_row = row + dx
                new_col = col + dy

                # If the neighbor is valid and hasn't been visited, explore it
                if is_valid_cell(new_row, new_col, check) and (new_col * 11 + new_row) not in visited:
                    if dfs(new_row, new_col, visited, target_row, target_col, check):
                        return True
            return False

        # Check for each starting position on the top and bottom sides
        for row in range(11):
            if board[row] == "0":
                visited = set()
                if dfs(row, 0, visited, 11, 10, "0"):
                    return True
        for col in range(0, 121, 11):
            if board[col] == "1":
                visited = set()
                if dfs(0, col // 11, visited, 10, 11, "1"):
                    return True
        return False  # No winner found

    def minimax(board, depth, current_player, first_player):
        other_player = "0" if current_player == "1" else "1"
        if has_winner(board):
            if current_player == "0":
                return -1, None  # blue won
            else:
                return 1, None  # red won

        if depth == 0:
            return 0, None  # no one won

        if current_player == "0":  # red, maximizing
            value = -10
            best = None
            for i in range(11 * 11):
                if board[i] is not None:
                    continue
                board[i] = current_player
                a, _ = minimax(board, depth - 1, other_player, first_player)
                board[i] = None
                if a > value:
                    value = a
                    best = i
                if value >= 1:
                    return value, best
                if current_player != first_player and value == 0:
                    return value, best
            return value, best
        else:  # blue, minimizing
            value = 10
            best = None
            for i in range(11 * 11):
                if board[i] is not None:
                    continue
                board[i] = current_player
                a, _ = minimax(board, depth - 1, other_player, first_player)
                board[i] = None
                if a < value:
                    value = a
                    best = i
                if value <= -1:
                    return value, best
                if current_player != first_player and value == 0:
                    return value, best
            return value, best

    def add_border(x, y, input_values, border_color):
        if x in [-1, 11] and y in [-1, 11]:
            input_values.append(0)
            return True
        if x not in [-1, 11] and y not in [-1, 11]:
            return False
        if x in [-1, 11]:
            input_values.append(1 if border_color else 0)
        else:
            input_values.append(0 if border_color else 1)
        return True

    def find_sure_win_move(board, player):
        new_board=board.copy()
        for depth in [1,3]:
            a, val = minimax(new_board, depth, player, player)
            if player == '0' and a > 0:
                return val
            elif player == '1' and a < 0:
                return val
        return None

    def runModel(cells, agent_is_blue):
        global session
        input_values = []
        board_size = 11

        if agent_is_blue:
            for x in range(-1, board_size + 1):
                for y in range(-1, board_size + 1):
                    if not add_border(x, y, input_values, 1):
                        id = PosToId(x, y)
                        input_values.append(1 if cells[id] == "1" else 0)

            for x in range(-1, board_size + 1):
                for y in range(-1, board_size + 1):
                    if not add_border(x, y, input_values, 0):
                        id = PosToId(x, y)
                        input_values.append(1 if cells[id] == "0" else 0)
        else:
            for y in range(-1, board_size + 1):
                for x in range(-1, board_size + 1):
                    if not add_border(x, y, input_values, 0):
                        id = PosToId(x, y)
                        input_values.append(1 if cells[id] == "0" else 0)

            for y in range(-1, board_size + 1):
                for x in range(-1, board_size + 1):
                    if not add_border(x, y, input_values, 1):
                        id = PosToId(x, y)
                        input_values.append(1 if cells[id] == "1" else 0)
            
        input_values2 = []
        for id in range((board_size + 2) * (board_size + 2)):
            input_values2.append(input_values[(board_size + 2) * (board_size + 2) - id - 1])
        
        for id in range((board_size + 2) * (board_size + 2)):
            input_values2.append(input_values[2 * (board_size + 2) * (board_size + 2) - id - 1])

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_data = np.array(input_values, dtype=np.float32).reshape(
            (1, 2, board_size + 2, board_size + 2))
        outputTensor = [tensor.flatten().tolist() for tensor in session.run([
            output_name], {input_name: input_data})][0]
        input_name2 = session.get_inputs()[0].name
        output_name2 = session.get_outputs()[0].name
        input_data2 = np.array(input_values2, dtype=np.float32).reshape(
            (1, 2, board_size + 2, board_size + 2))
        outputTensor2 = [tensor.flatten().tolist() for tensor in session.run([
            output_name2], {input_name2: input_data2})][0]
        average_output = []
        for id in range(board_size * board_size):
            average_output.append(
                (outputTensor[id] + outputTensor2[board_size * board_size - id - 1]) / 2)
        final_output = []
        if agent_is_blue:
            # Transpose if agent is blue
            for x in range(board_size):
                for y in range(board_size):
                    id = PosToId(x, y)
                    final_output.append(average_output[id])
        else:
            final_output = average_output
        
        return final_output
            
    ai_board=matrix.copy()
    sure_win_move=find_sure_win_move(ai_board,agent)
    if sure_win_move:
        print("Agent can surely win by move",sure_win_move)
        return sure_win_move
    result = runModel(ai_board, agent_is_blue)
    best = -1
    max_rating = float('-inf')
    for i in range(len(matrix)):
        if matrix[i] is None:
            if best == -1 or result[i] > max_rating:
                best = i
                max_rating = result[i]


    test_board = matrix.copy()
    test_board[best] = agent

    sure_win = find_sure_win_move(test_board, player)
    if sure_win is not None:
        print(f"Player can surely win with the suggested move {sure_win}")
        print("Blocking the player")
        best = sure_win

    return best
    


@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return error(request,exc.status_code, "Oops! This route does not exist.")


@app.post('/getAImove')
def get_ai_move(data: InputData):
    try:
        matrix = data.matrix
        player = data.player
        agent = data.agent
        agent_is_blue = data.agent_is_blue

        # Call your handle_click function or run inference directly here
        next_move = handle_click(matrix, player, agent,agent_is_blue)

        return {"nextMove": next_move}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing request")


@app.get("/")
async def welcome_user(request: Request, user: str = "user"):
    session_id = request.cookies.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())
    response = FileResponse("public/html/index.html")
    response.set_cookie(key="session_id", value=session_id)
    asyncio.create_task(clear_dicts())
    return response

def error(request:Request,Code:int,Info:str):
    response = templates.TemplateResponse(
        "error.html", {"request": request,"Code":Code,"Detail": Info})
    return response


@app.get('/favicon.ico')
async def get_icon():
    return FileResponse('public/img/favicon.png')

@app.get("/new-game/{game_id}")
async def new_game(
    request: Request,
    game_id: str,
    AI_MODE: Optional[bool] = Query(None)
):
    if game_id not in game_mapping:
        return error(request, 404, "Game with specified Id does not exist")
    session_id = request.cookies.get("session_id")
    if session_id in cur_games and cur_games[session_id]==game_id:
        return RedirectResponse('/')
    response = templates.TemplateResponse(
        "hex.html", {"request": request, "AI_MODE": AI_MODE})
    id = game_mapping[game_id][0]
    response.set_cookie(key="session_id", value=id)
    response.set_cookie(key="game_id", value=game_id)
    return response


@app.get("/create-game")
async def create_game(request: Request) -> Dict[str, str]:
    """
    Create a new game with a unique game ID and generate two session IDs.
    """
    game_id = str(uuid.uuid4())  # Generate a unique game ID
    # Generate session ID for player 1
    session_id1 = request.cookies.get("session_id")
    if session_id1 is None:
        session_id1 = str(uuid.uuid4())
    session_id2 = str(uuid.uuid4())  # Generate session ID for player 2
    # Store the mapping between game ID and session IDs
    game_mapping[game_id] = [session_id1,session_id2]
    game_coroutines[game_id] = asyncio.Event()
    return {"id":game_id}

@app.get('/watch/{game_id}')
def watch_game(request: Request,game_id:str):
    if game_id not in start_times:
        return error(request, 404, "Game with specified Id does not exist")
    try:
        name1,name2=game_participants[game_id]
        response = templates.TemplateResponse(
            "watchHex.html", {"request": request, "player1":name1, "player2":name2})
        id = request.cookies.get("session_id")
        response.set_cookie(key="session_id", value=id)
        response.set_cookie(key="game_id", value=game_id)
        return response
    except Exception:
        return error(request, 520, "Some Internal Error! try again.")

@app.get('/start-game/{game_id}')
async def start(request:Request,game_id: str):
    if game_id not in game_mapping:
        return error(request,400, "Game with specified Id does not exist")
    if game_id in game_coroutines:
        # Await the game coroutine
        event=game_coroutines[game_id]
        await event.wait()
        print("resolved")
        return {'start': True}
    else:
        print("no")
        return error(request, 404, "Game not found")

    

@app.get("/join-game/{game_id}")
async def join_game(game_id: str,request: Request) -> str:
    """
    Join the game with the provided game ID using the given session ID.
    """
    if game_id not in game_mapping:
        return error(request, 400, "Game with specified Id does not exist")
    id2=game_mapping[game_id][1]
    if id2 in cur_games and cur_games[id2]==game_id:
        return error(request, 400, "Game already has two players")
    response = templates.TemplateResponse(
        "hex.html", {"request": request, "AI_MODE":False})
    response.set_cookie(key="session_id", value=id2)
    response.set_cookie(key="game_id", value=game_id)
    return response


def add_name(game_id,user_name):
    if game_id not in game_participants:
        game_participants[game_id]=[]
    game_participants[game_id].append(user_name)
    

@app.websocket("/ws/{game_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, session_id: str):
    global socket_list
    global connections
    global start_times
    await websocket.accept()
    if session_id == game_mapping[game_id][1]:
        event = game_coroutines[game_id]
        event.set()
    if game_id not in connections:
        connections[game_id] = {}
    if game_id not in socket_list:
        socket_list[game_id]=[]
    connections[game_id][session_id] = websocket
    socket_list[game_id].append(websocket)
    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data) 
            if message.get("action") == "gameData":
                id=message.get('answer_for')
                name1,name2=game_participants.get(game_id,[None,None])
                message["Type"]= 1
                message["p1_name"]=name1
                message["p2_name"]=name2
                socket_to=watch_list[game_id][id]
                try:
                    await socket_to.send_text(json.dumps(message))
                except Exception as e:
                    print("Error::",e)
                    
            elif message.get("action") == "blue":
                name1,name2=game_participants.get(game_id,[None,None])
                if name1!=message.get('name'):
                    game_participants[game_id]=[name2,name1]
            elif message.get('action')=='moveDetails':
                message["Type"]=2
                await pass_message(game_id,message)
                await broadcast(game_id,message,not message.get("player"))
            elif message.get('action')=='missed':
                message["Type"]=3
                await pass_message(game_id,message)
                await broadcast(game_id,message,not message.get("player"))
            elif message.get('action')=='switch':
                if message.get("player"):
                    continue
                message["Type"]=4
                name1,name2=game_participants[game_id]
                game_participants[game_id]=[name2,name1]
                message["player1"]=name2
                message["player2"]=name1
                if len(socket_list[game_id])>1:
                    sock1,sock2=socket_list[game_id]
                    socket_list[game_id]=[sock2,sock1]
                    message['turn']=False
                    message['player']=True
                    sock1,sock2=socket_list[game_id]
                    await sock1.send_text(json.dumps(message))
                    message['turn']=True
                    message['player']=False
                    await sock2.send_text(json.dumps(message))
                await broadcast(game_id,message,False)
            elif message.get("action") == "acknowledge":
                user_name = message.get("name")
                cur_games[session_id] = game_id
                add_name(game_id,user_name)
                if message.get("AI"):
                    add_name(game_id,"AI")
                    start_times[game_id]=datetime.now()
                await handle_acknowledgment(game_id)
            elif message.get('action')=='WIN':
                message["Type"]=5
                name1,name2=game_participants.get(game_id)
                name= name1 if message.get("player1") else name2
                message['Winner']=name
                await broadcast(game_id,message)
            elif message.get("action") == 'END':
                message["Type"]=5
                name1,name2=game_participants.get(game_id)
                name= name1 if message.get("player1") else name2
                message['Winner']=name
                await broadcast(game_id,message)
                connections.pop(game_id,None)
                game_mapping.pop(game_id,None)
        except WebSocketDisconnect:
            # WebSocket connection closed
            try:
                start_times.pop(game_id,None)
                turn_details.pop(session_id,None)
                turn_counts.pop(game_id,None)
                other= 1 if game_mapping[game_id][0]==session_id else 0
                other_session=game_mapping[game_id][other]
                if game_id in connections and other_session in connections[game_id]:
                    socket=connections[game_id][other_session]
                    message = json.dumps(
                        {"Type": 5, "win":True})
                    await socket.send_text(message)
                    connections.pop(game_id,None)
                    game_mapping.pop(game_id,None)
            except Exception as e:
                print(f"WebSocket connection closed with exception: {e} 1")
            break
        except Exception as e:
            print(f"WebSocket connection closed with exception: {e} 2")
            break
            


async def pass_message(game_id,message):
    if game_id not in socket_list or len(socket_list[game_id])<=1:
        return
    
    sock1,sock2=socket_list[game_id]
    message['turn']=True
    if message.get("player"):
        await sock2.send_text(json.dumps(message))
        message['turn']=False
        await sock1.send_text(json.dumps(message))
    else:
        await sock1.send_text(json.dumps(message))
        message['turn']=False
        await sock2.send_text(json.dumps(message))


@app.post('/disconnect')
async def disconnect(data:Disc_data):
    data=data.model_dump()
    game_id=data.get("id")
    name=data.get("name")
    name1,name2=game_participants.get(game_id,[None,None])
    Winner=name1 if name2==name else name2
    message={"Type":5,"Winner":Winner}
    await broadcast(game_id,message)
    return {"success":True}


@app.websocket("/ws/watch/{game_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, session_id: str):
    global watch_list
    await websocket.accept()
    
    if game_id not in watch_list:
        watch_list[game_id] = {}
    watch_list[game_id][session_id] = websocket

    try:
        while True:
            data = await websocket.receive_text()  # Wait for incoming messages
            # Handle incoming data here
            print(f"Received data: {data}")
            
            # Optionally, send a response back to the client
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id} and game {game_id}")
        # Clean up the watch list entry if the WebSocket disconnects
        watch_list.get(game_id,{}).pop(session_id,None)
        if not len(watch_list[game_id]):
            watch_list.pop(game_id,None)


async def handle_acknowledgment(game_id: str):
    s1,s2=game_mapping[game_id]
    if s1 in cur_games and s2 in cur_games and len(connections.get(game_id, {})) == 2:
        await start_game(game_id)


async def start_game(game_id: str):
    global socket_list
    global turn_counts
    global turn_details
    try:
        if len(connections[game_id].values()) != 2:
            return False

        s1, s2 = game_mapping[game_id]
        sock1, sock2 = socket_list[game_id]

        # Randomly choose turn
        turn = random.randint(0, 1)
        if turn==0 :
            turn_details[s1]=True
        else:
            turn_details[s2]=True
            socket_list[game_id]=[sock2,sock1]
        
        # Send start message to sockets
        name1,name2=game_participants[game_id]
        start_message1 = {"Type":1,"readyToStart":True,"turn":True,"p1":name1,"p2":name2}
        start_message2 = {"Type":1,"readyToStart":True,"turn":False,"p1":name1,"p2":name2}

        sock1, sock2 = socket_list[game_id]
        # Define a timeout for waiting for responses
        start_times[game_id]=datetime.now()
        await sock1.send_text(json.dumps(start_message1))
        await sock2.send_text(json.dumps(start_message2))
        turn_counts[game_id]=0
    except:
        print("Error")
        

@app.on_event("startup")
def start_up():
    global session
    global app
    global templates
    current=os.path.dirname(os.path.abspath(__file__))
    os.chdir(current)
    session=onnxruntime.InferenceSession(r"./AI.onnx")
    templates=Jinja2Templates(directory="public/html/")
    app.mount("/public", StaticFiles(directory="public"), name="public")
    print("---------------------------")
    print("session loaded",session is not None)
    print("---------------------------")
    if session is None:
        exit(1)
        
@app.get('/get-live-games')
async def get_live_games():
    games=[]
    
    for game_id in start_times:
        try:
            time_elapsed=datetime.now()-start_times[game_id]
            name1,name2=game_participants[game_id]
            game={
                "player1":name1,
                "player2":name2,
                "duration":time_elapsed,
                "gameId":game_id,
            }
            games.append(game)
        except Exception as e:
            print("error")
        
    return {"data":games}


@app.post('/get_game_state')
async def get_game_info(request:Request,data:gameRequest):
    game_id=data.id
    for socket in connections[game_id].values():
        try:
            message = json.dumps({"Type": 6, "id":request.cookies.get("session_id")})
            await socket.send_text(message)
        except Exception:
            pass
    return {"success":True}
    
async def broadcast(game_id,message,blue_turn=False):
    if game_id not in watch_list:
        return
    message['blue']=blue_turn
    for socket in watch_list[game_id].values():
        try:
            await socket.send_text(json.dumps(message))
        except Exception as e:
            watch_list.pop(game_id,None)
    if message.get("Type")==5:
        watch_list.pop(game_id,None)


async def clear_dicts():
    global underway
    if underway:
        return
    print("Cleaning started")
    underway = True

    # Take a snapshot of the dictionaries before sleeping
    snapshot_mapping = dict(game_mapping)
    snapshot_connections = dict(connections)
    snapshot_coroutines = dict(game_coroutines)
    snapshot_cur_games = dict(cur_games)
    snapshot_turn_counts = dict(turn_counts)
    snapshot_turn_details = dict(turn_details)
    snapshot_start_times = dict(start_times)
    snapshot_game_participants=dict(game_participants)
    snapshot_watch_list=dict(watch_list)
    snapshot_socket_list=dict(socket_list)

    await asyncio.sleep(15*60)  # 15 minutes in seconds

    # Compare with the snapshot and clear any new items
    for d, snapshot in zip([game_mapping, connections, game_coroutines, cur_games, turn_counts, turn_details,watch_list,game_participants,start_times,socket_list],
                           [snapshot_mapping, snapshot_connections, snapshot_coroutines, snapshot_cur_games,
                            snapshot_turn_counts, snapshot_turn_details,snapshot_watch_list,snapshot_game_participants,snapshot_start_times,snapshot_start_times,snapshot_socket_list]):
        for key in list(snapshot.keys()):
                d.pop(key,None)

    print("Dictionaries cleaned")
    underway = False

@app.get("/health_check")
def health_path():
    return {"Status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app,port=9000)
