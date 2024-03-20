from asyncio import Future
import json
import random
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi import Query
import uuid
import os
import uvicorn
# Import pymongo
import pymongo
from datetime import datetime
from typing import Union, Dict, List, Set, Optional

import string


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
acknowledgements:Dict[str,Set[str]] = {}
connections: Dict[str, Dict[str,WebSocket]] = {}
game_coroutines = {}
game_state:Dict[str,List[List[int]]]={}


@app.middleware("http")
async def redirect_new_game(request: Request, call_next):
    if request.url.path.startswith("/new-game/public"):
        # Extract the part after "/public"
        new_path = request.url.path.split("/public", 1)[-1]
        # Redirect to "/public/whatever_was_there"
        print(f"/public{new_path}")
        return RedirectResponse(url=f"/public{new_path}")
    return await call_next(request)

@app.get("/")
async def welcome_user(request: Request, user: str = "user"):
    session_id = request.cookies.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())
    response = FileResponse("public/html/index.html")
    response.set_cookie(key="session_id", value=session_id)
    return response


@app.get("/new-game/{game_id}")
async def new_game(
    request: Request,
    game_id: str,
    AI_MODE: Optional[bool] = Query(None)
):
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
    cur_games[session_id1]=game_id
    future = Future()
    game_coroutines[game_id] = future
    return {"id":game_id}


@app.get('/start-game/{game_id}')
async def start(game_id: str):
    print(game_id)
    # Check if the game coroutine exists
    if game_id in game_coroutines:
        # Await the game coroutine
        await game_coroutines[game_id]
        print("resolved")
        return {'start': True}
    else:
        print("no")
        raise HTTPException(status_code=404, detail="Game not found")

    

@app.get("/join-game/{game_id}")
async def join_game(game_id: str,request: Request) -> str:
    """
    Join the game with the provided game ID using the given session ID.
    """
    s=[str(i) for i in game_mapping.keys()]
    game_id=s[-1]
    id2=game_mapping[game_id][1]
    if id2 in cur_games and cur_games[id2]==game_id:
        raise HTTPException(
            status_code=400, detail="Game already has two players")
    response = templates.TemplateResponse(
        "hex.html", {"request": request, "AI_MODE":False})
    response.set_cookie(key="session_id", value=id2)
    response.set_cookie(key="game_id", value=game_id)
    cur_games[id2]=game_id
    
    # Resolve the future object
    return response


@app.websocket("/ws/{game_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, session_id: str):
    await websocket.accept()
    if session_id==game_mapping[game_id][1]:
        game_coroutines[game_id].set_result(None)
    if game_id not in connections:
        connections[game_id] = {}
    try:
        while True:
            # Receive message from client (acknowledgment)
            data = await websocket.receive_text()
            print(data)
            if data == '{"action":"acknowledge"}':
                connections[game_id][session_id]=websocket
                await handle_acknowledgment(game_id)
    except Exception as e:
        print(f"WebSocket connection closed with exception: {e}")
    

async def handle_acknowledgment(game_id: str):
    if len(connections.get(game_id, {})) == 2:
        await start_game(game_id)


async def start_game(game_id: str):
    if (len(connections[game_id].values())!=2):
        return False
    temp=[]
    for i in range(11):
        temp.append([0 for i in range(11)])
    game_state[game_id]=temp
    s1,s2=game_mapping[game_id]
    sock1,sock2=connections[game_id][s1],connections[game_id][s2]
    turn = random.randint(0, 1)
    if(turn==1):
        sock1, sock2 = connections[game_id][s2], connections[game_id][s1]
    await sock1.send_text('{"Type":1,"readyToStart": true,"turn":true}')
    await sock2.send_text('{"Type":1,"readyToStart": true,"turn":false}')
    
    
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
        raise HTTPException(
            status_code=400, detail="Game ID or session ID not found in cookies")

    # Retrieve player number based on session_id
    
    if player_number is None:
        raise HTTPException(
            status_code=400, detail="Player number not found for session ID")

    # Retrieve game state (matrix) for the given game_id
    game = game_state.get(game_id)
    if game is None:
        raise HTTPException(
            status_code=404, detail="Game state not found for game ID")
    # Update the game state with the player's move
    flag=False
    if (row < 0 or column < 0):
        flag=False
    else:
        if(game[row][column]!=0):
            raise HTTPException(
                status_code=404, detail="Invalid move")
        game[row][column] = player_number+1
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
    return {"message": "Move successfully made"}


if __name__ == "__main__":
    uvicorn.run(app,port=9000)
