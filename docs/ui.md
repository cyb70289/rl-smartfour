smart-four UI
=============
A web UI can be accessed local or remote. Play directly in browser. Can play against person or machine.
- in 3D style, can rotate to view from different angles, as pieces may block each other and not visible
- draw pieces as round bucket, place with bottom up, can stack on board positions and on another piece's bottom
- white and black pieces, white player first, show number of pieces left and current player in ui
- highlight last move
- can revert last move, just one move, not earlier
- if win, highlight the line and show winner, show draw when all pieces are placed and no winner
- can play with person or machine

play with person
----------------
- "revert" can only happen immediately after move. e.g., player1 moves, player2 moves, player2 can now revert its last move, but player1 cannot.

play with machine
-----------------
- machine move is based on an alphazero like model trained by this project, served locally
- machine move may take time, disable related ui controls during model thinking
- can disable or set think effort in UI, related to mtcs searches: disable (policy only), effort (mtcs search steps)
- since machine moves immediately after human move, the "revert" shall revert both the machine and last human moves, to the state that human can retry last move
