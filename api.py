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

import string
underway=False

class Move(BaseModel):
    row:int
    column:int

class User(BaseModel):
    name: str
    profile_pic: str

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


@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return error(request,exc.status_code, "Oops! This route does not exist.")


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
    await asyncio.sleep(15*60)  # 15 minutes in seconds
    for d in [game_mapping, connections, game_coroutines, cur_games,turn_counts,turn_details]:
        d.clear()
    print("Dictionaries cleaned")
    underway = False

@app.get("/health_check")
def health_path():
    return {"Status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app,port=9000)
