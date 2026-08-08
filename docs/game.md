smart-four game rules
=====================
Smart-four is an advanced tic-tac-toe game in 5x5x5 3D space. Two player place pieces in turn, and win by lining four same colored pieces in the 3D space.


board and piece
---------------
Two player (white, black), each has 32 pieces. Pieces can be stacked up. Board is 5x5 size, i.e, can hold 25 pieces.
Piece can place in empty slots on board, or stack on another piece (from any player). At most 5 pieces can be stack up.
So, the total space in 5x5x5. Obviously, a piece can only stack on an existing piece, cannot "float" in the air.


how to win
----------
Any player lines 4 pieces first win the game
- lines in same horizontal plane, note there are 5 horizontal spaces, and line can be diagonal
- lines in same vertical plane, e.g., 4 pieces in one stack, or a stepping style line going up, note both the plane and line can be diagonal

It's a draw if all pieces are placed without a winner.
