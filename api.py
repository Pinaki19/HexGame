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

import string
underway=False

session = onnxruntime.InferenceSession(r"./AI.onnx")

class Move(BaseModel):
    row:int
    column:int

class User(BaseModel):
    name: str
    profile_pic: str
    

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


app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/html/")

game_mapping: Dict[str, List[str]] = {}
cur_games: Dict[str,str] = {}
connections: Dict[str, Dict[str,WebSocket]] = {}
game_coroutines = {}
turn_counts={}
turn_details={}


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
    print(game_mapping)
    print(cur_games)
    print(connections)
    print(game_coroutines)
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


@app.websocket("/ws/{game_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, session_id: str):
    await websocket.accept()
    if session_id == game_mapping[game_id][1]:
        event = game_coroutines[game_id]
        event.set()
    if game_id not in connections:
        connections[game_id] = {}
    connections[game_id][session_id] = websocket
    try:
        while True:
            # Receive message from client (acknowledgment)
            data = await websocket.receive_text()
            print(data)
            if data == '{"action":"acknowledge"}':
                cur_games[session_id] = game_id
                await handle_acknowledgment(game_id)
            if data == '{"action":"END"}':
                connections.pop(game_id,None)
                game_mapping.pop(game_id,None)
                break
    except WebSocketDisconnect:
        # WebSocket connection closed
        try:
            turn_details.pop(session_id,None)
            turn_counts.pop(game_id,None)
            other= 1 if game_mapping[game_id][0]==session_id else 0
            other_session=game_mapping[game_id][other]
            if game_id in connections and other_session in connections[game_id]:
                socket=connections[game_id][other_session]
                print("sock: ",socket)
                message = json.dumps(
                    {"Type": 5, "win":True})
                await socket.send_text(message)
                connections.pop(game_id,None)
                game_mapping.pop(game_id)
        except Exception as e:
            print(f"WebSocket connection closed with exception: {e} 1")
        # Perform cleanup or other actions as needed
    except Exception as e:
        print(f"WebSocket connection closed with exception: {e} 2")


async def handle_acknowledgment(game_id: str):
    s1,s2=game_mapping[game_id]
    if s1 in cur_games and s2 in cur_games and len(connections.get(game_id, {})) == 2:
        await start_game(game_id)


async def start_game(game_id: str):
    try:
        if len(connections[game_id].values()) != 2:
            return False

        s1, s2 = game_mapping[game_id]
        sock1, sock2 = connections[game_id][s1], connections[game_id][s2]

        # Randomly choose turn
        turn = random.randint(0, 1)
        if not turn:
            turn_details[s1]=True
        else:
            turn_details[s2]=True
        if turn == 1:
            sock1, sock2 = sock2, sock1  # Swap sockets

        # Send start message to sockets
        start_message1 = '{"Type":1,"readyToStart":true,"turn":true}'
        start_message2 = '{"Type":1,"readyToStart":true,"turn":false}'

        # Define a timeout for waiting for responses
    
        await sock1.send_text(start_message1),
        await sock2.send_text(start_message2)
        turn_counts[game_id]=0
    except:
        print("Error")
        
    
    
@app.post("/make_move")
async def make_move(request: Request, data:Move):
    # Retrieve game_id and session_id from cookies
    data=data.dict()
    row=data.get("row",-1)
    column=data.get("column",-1)
    game_id = request.cookies.get("game_id")
    session_id = request.cookies.get("session_id")
    player_number = 0 if game_mapping[game_id][0] == session_id else 1
    for session_id in game_mapping[game_id]:
        if session_id not in connections[game_id]:
            return {"message": "waiting for other player"}
    if not game_id or not session_id:
        return error(request, 400, "Game ID or session ID not found in cookies")

    # Retrieve player number based on session_id
    
    if player_number is None:
        return error(request, 400, "Player number not found for session ID")

    flag=False
    if (row < 0 or column < 0):
        flag=False
    else:
        flag=True
    t= 0 if player_number==1 else 1
    sock1 = connections[game_id][game_mapping[game_id][t]]
    sock2 = connections[game_id][game_mapping[game_id][(t+1)%2]]
    
    message1 = json.dumps({"Type": 2, "turn": True})
    message2 = json.dumps({"Type": 2, "turn": False})
    if(flag):
        message1 = json.dumps(
            {"Type": 0, "row": row, "column": column, "turn": True})
        message2 = json.dumps(
            {"Type": 0, "row": row, "column": column, "turn": False})
    
    await sock1.send_text(message1)
    await sock2.send_text(message2)
    turn_counts[game_id]+=1
    return {"message": "Move successfully made"}

@app.get('/swith_player')
async def switch_player(request:Request):
    game_id = request.cookies.get("game_id")
    session_id = request.cookies.get("session_id")
    print(turn_details,turn_counts)
    s1,s2=game_mapping[game_id]
    if s1!=session_id:
        s1,s2=s2,s1
    if turn_counts[game_id]==1 and session_id not in turn_details:
        sock1 = connections[game_id][s1]
        sock2 = connections[game_id][s2]
        message1 = json.dumps({"Type": 3, "starting": True,"turn":False})
        message2 = json.dumps({"Type": 3, "starting": False,"turn":True})
        await sock1.send_text(message1)
        await sock2.send_text(message2)
    return {"message": "swith request processed"}


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

    await asyncio.sleep(15*60)  # 15 minutes in seconds

    # Compare with the snapshot and clear any new items
    for d, snapshot in zip([game_mapping, connections, game_coroutines, cur_games, turn_counts, turn_details],
                           [snapshot_mapping, snapshot_connections, snapshot_coroutines, snapshot_cur_games,
                            snapshot_turn_counts, snapshot_turn_details]):
        for key in list(snapshot.keys()):
                d.pop(key,None)

    print("Dictionaries cleaned")
    underway = False

@app.get("/health_check")
def health_path():
    return {"Status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app,port=9000)
