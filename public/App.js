const url = "https://hex-b5w.pages.dev/11_2w4_2000.onnx";
let info = "";
let max_rating;
const sess = new onnx.InferenceSession();

async function load_model() {
    await sess.loadModel(url);
    return true;
}

function PosToId(x, y) {
    return x + board_size * y;
}

function IdToPos(id) {
    const x = id % board_size;
    const y = (id - x) / board_size;
    return [x, y];
}

// Assuming boardSize is 11x11 and the board is represented as a 1D array of size 121
function HasWinner(board) {
    // Define directions for hexagon grid
    const directions = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];

    // Function to check if a cell is valid
    const isValidCell = (row, col, check) => {
        return row >= 0 && row < 11 && col >= 0 && col < 11 && board[col * 11 + row] === check;
    };

    // Function to perform DFS traversal
    const dfs = (row, col, visited, targetRow, targetCol, check) => {
        visited.add(col * 11 + row);
        //console.log(visited);
        // If we reach the opposite side, we have a winner
        if ((targetRow === 10 && row === targetRow) || (targetCol === 10 && col === targetCol)) {
            winner=check;
            return true;
        }

        // Check all neighbors
        for (const [dx, dy] of directions) {
            const newRow = row + dx;
            const newCol = col + dy;

            // If the neighbor is valid and hasn't been visited, explore it
            if (isValidCell(newRow, newCol, check) && !visited.has(newCol * 11 + newRow)) {
                if (dfs(newRow, newCol, visited, targetRow, targetCol, check)) {
                    return true;
                }
            }
        }
        return false;
    };

    // Check for each starting position on the top and bottom sides
    for (let row = 0; row < 11; row++) {
        if (board[row] === "0") {
            const visited = new Set();
            if (dfs(row, 0, visited, 11, 10, "0")) {
                return true;
            }
        }
    }
    for (let col = 0; col < 121; col += 11) {
        if (board[col] === "1") {
            const visited = new Set();
            if (dfs(0, col / 11, visited, 10, 11, "1")) {
                return true;
            }
        }
    }

    return false; // No winner found
}




function minimax(board, depth, current_player, first_player) {
    const other_player = current_player === "0" ? "1" : "0";
    if (HasWinner(board)) {
        
        if (current_player === "0") {
            return [-1, null]; // blue won
        } else {
            return [1, null]; // red won
        }
    }

    if (depth === 0) {
        return [0, null]; // no one won
    }
    if (current_player === "0") { // red, maximizing
        let value = -10;
        let best = null;
        for (let i = 0; i < board_size * board_size; i++) {
            if (board[i] !== null) {
                continue;
            }
            board[i] = current_player;
            const a = minimax(board, depth - 1, other_player, first_player);
            board[i] = null;
            if (a[0] > value) {
                value = a[0];
                best = i;
            }
            if (value >= 1) {
                // this is good enough
                return [value, best];
            }
            if (current_player !== first_player && value === 0) {
                // this is good enough for the second player.
                return [value, best];
            }
        }
        return [value, best];
    } else { // blue, minimizing
        let value = 10;
        let best = null;
        for (let i = 0; i < board_size * board_size; i++) {
            if (board[i] !== null) {
                continue;
            }
            board[i] = current_player;
            const a = minimax(board, depth - 1, other_player, first_player);
            board[i] = null;
            if (a[0] < value) {
                value = a[0];
                best = i;
                if (value <= -1) {
                    // this is bad enough
                    return [value, best];
                }
                if (current_player !== first_player && value === 0) {
                    // this is good enough for the second player.
                    return [value, best];
                }
            }
        }
        return [value, best];
    }
}

function AddBorder(x, y, input_values, border_color) {
    if ([-1, board_size].includes(x) && [-1, board_size].includes(y)) {
        input_values.push(0);
        return true;
    }
    if (!([-1, board_size].includes(x) || [-1, board_size].includes(y))) {
        return false;
    }
    if ([-1, board_size].includes(x)) {
        input_values.push(border_color ? 1 : 0);
    } else {
        input_values.push(border_color ? 0 : 1);
    }
    return true;
}

function findSureWinMove(board, player) {
    for (let depth of [1, 3]) {
        const a = minimax(board, depth, player, player);
        if (player === '0') {
            if (a[0] > 0) {
                return a[1];
            }
        } else {
            if (a[0] < 0) {
                return a[1];
            }
        }
    }

    return null;
}

async function handle_click(matrix, player, agent, ai_board,agent_is_blue){
    return new Promise((resolve, reject) => {
        const sure_win_move = findSureWinMove(ai_board, agent);
        if (sure_win_move !== null) {
            console.log("Agent can surely win with the suggested move", sure_win_move);
            updateTurn(200, `Agent can surely win with the suggested move ${sure_win_move}`);
            matrix[sure_win_move] = agent;
            set_color(sure_win_move, true,false);
            resolve(sure_win_move);
            return;
        }
        runModel(ai_board,agent_is_blue).then((result) => {
            let best = -1;
            for (let i = 0; i < board_size * board_size; i++) {
                if (matrix[i] === null) {
                    if (best === -1 || result[i] > max_rating) {
                        best = i;
                        max_rating = result[i];
                    }
                }
            }
            let score_sum = 0;
            for (let i = 0; i < board_size * board_size; i++) {
                if (matrix[i] === null) {
                    const score = Math.pow(2, result[i] - max_rating);
                    score_sum += score;
                }
            }
            let test_board = Array.from(matrix);
            test_board[best] = agent;
            const sure_win = findSureWinMove(test_board, player);
            if (sure_win !== null) {
                updateTurn(250, `Player can surely win with the suggested move ${sure_win}`);
                console.log("Player can surely win with the suggested move",sure_win, "Blocking the player");
                best = sure_win;
            }
            resolve(best);
        }).catch(reject);
    });
}

async function runModel(cells,agent_is_blue) {
    return new Promise(async (resolve, reject) => {
        try {
            info = "waiting for agent to move...";
            let input_values = [];
            if (agent_is_blue) {
                for (let x = -1; x < board_size + 1; x++) {
                    for (let y = -1; y < board_size + 1; y++) {
                        if (!AddBorder(x, y, input_values, 1)) {
                            const id = PosToId(x, y);
                            input_values.push(cells[id] === "1" ? 1 : 0);
                        }
                    }
                }
                for (let x = -1; x < board_size + 1; x++) {
                    for (let y = -1; y < board_size + 1; y++) {
                        if (!AddBorder(x, y, input_values, 0)) {
                            const id = PosToId(x, y);
                            input_values.push(cells[id] === "0" ? 1 : 0);
                        }
                    }
                }
            } else {
                for (let y = -1; y < board_size + 1; y++) {
                    for (let x = -1; x < board_size + 1; x++) {
                        if (!AddBorder(x, y, input_values, 0)) {
                            const id = PosToId(x, y);
                            input_values.push(cells[id] === "0" ? 1 : 0);
                        }
                    }
                }
                for (let y = -1; y < board_size + 1; y++) {
                    for (let x = -1; x < board_size + 1; x++) {
                        if (!AddBorder(x, y, input_values, 1)) {
                            const id = PosToId(x, y);
                            input_values.push(cells[id] === "1" ? 1 : 0);
                        }
                    }
                }
            }

            let input_values2 = [];
            for (let id = 0; id < (board_size + 2) * (board_size + 2); id++) {
                input_values2.push(input_values[(board_size + 2) * (board_size + 2) - id - 1]);
            }
            for (let id = 0; id < (board_size + 2) * (board_size + 2); id++) {
                input_values2.push(input_values[2 * (board_size + 2) * (board_size + 2) - id - 1]);
            }

            const outputTensor = await evalModel(input_values);
            const outputTensor2 = await evalModel(input_values2);
    
            let average_output = [];
            for (let id = 0; id < board_size * board_size; id++) {
                average_output.push((outputTensor[id] + outputTensor2[board_size * board_size - id - 1]) / 2);
            }
            let final_output = [];
            if (agent_is_blue) {
                // need to transpose
                for (let x = 0; x < board_size; x++) {
                    for (let y = 0; y < board_size; y++) {
                        const id = PosToId(x, y);
                        final_output.push(average_output[id]);
                    }
                }
            } else {
                final_output = average_output;
            }
        
            resolve(final_output);
        } catch (e) {
            reject(e);
        }
    });
}


async function evalModel(input_array) {
    const inputTensor = new onnx.Tensor(new Float32Array(input_array), 'float32', [1, 2, board_size + 2, board_size + 2]);
    const outputMap = await sess.run([inputTensor]);
    const outputTensor = outputMap.values().next().value;
    const outputData = outputTensor.data;
    return outputData;
}
