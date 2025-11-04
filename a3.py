# CMPUT 455 Assignment 3 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas

import sys
import signal

class CommandInterface:
    # The following is already defined and does not need modification
    # However, you may change or add to this code as you see fit, e.g. adding class variables to init

    def __init__(self):
        # Define the string to function command mapping
    
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,   # init_game w h p s [board]
            "show"     : self.show,
            "timelimit": self.timelimit,   # timelimit seconds
            "load_patterns"    : self.load_patterns,       # see assignment spec
            "policy_moves"    : self.policy_moves,       # see assignment spec
            "position_evaluation": self.position_evaluation, # see assignment spec
            "move_evaluation": self.move_evaluation, # see assignment spec
            "score"    : self.score,
            "play" : self.play
        }

        # Game state
        self.board = [[None]]
        self.player = 1           # 1 or 2
        self.handicap = 0.0       # P2’s handicap
        self.score_cutoff = float("inf")

        # This variable keeps track of the maximum allowed time to solve a position
        self.timelimit = 1
        
        self.patterns = [] # you may change this

    # Convert a raw string to a command and a list of arguments
    def process_command(self, s):
        
        class TimeoutException(Exception):
            pass
        
        def handler(signum, frame):
            raise TimeoutException("Function timed out.")
        
        s = s.lower().strip()
        if len(s) == 0:
            return True
        command = s.split(" ")[0]
        args = [x for x in s.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Uknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timelimit)
            
            return self.command_dict[command](args)
        except TimeoutException:
            print(f"Command '{s}' timed out after {self.timelimit} seconds.", file=sys.stderr)
            print("= -1\n")
            return False
        except Exception as e:
            print("Command '" + s + "' failed with exception:", file=sys.stderr)
            print(e, file=sys.stderr)
            print("= -1\n")
            return False
        finally: 
            signal.alarm(0)

        
    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    # Helper function for command argument checking
    # Will make sure there are enough arguments, and that they are valid integers
    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Recieved arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr)
                    return False
        return True


    # init_game w h p s [board_str]
    # Note that your init_game function must support initializing the game with a string (this was not necessary in A1).
    # We already have implemented this functionality in our provided init_game function.
    def init_game(self, args):
        # Check arguments
        if len(args) > 4:
            self.board_str = args.pop()
        else:
            self.board_str = ""
        if not self.arg_check(args, "w h p s"):
            return False
        w, h, p, s = args
        if not (1 <= w <= 10000 and 1 <= h <= 10000):
            print("Invalid board size:", w, h, file=sys.stderr)
            return False
        
        #Initialize game state

        self.width = w
        self.height = h
        self.handicap = p
        if s == 0:
            self.score_cutoff = float("inf")
        else:
            self.score_cutoff = s
        
        self.board = []
        for r in range(self.height):
            self.board.append([0]*self.width)
        self.player = 1

        # optional board string to initialize the game state
        if len(self.board_str) > 0:
            board_rows = self.board_str.split("/")
            if len(board_rows) != self.height:
                print("Board string has wrong height.", file=sys.stderr)
                return False
            
            p1_count = 0
            p2_count = 0
            for y, row_str in enumerate(board_rows):
                if len(row_str) != self.width:
                    print("Board string has wrong width.", file=sys.stderr)
                    return False
                for x, c in enumerate(row_str):
                    if c == "1":
                        self.board[y][x] = 1
                        p1_count += 1
                    elif c == "2":
                        self.board[y][x] = 2
                        p2_count += 1
            
            if p1_count > p2_count:
                self.player = 2
            else:
                self.player = 1

        self.timelimit = 1

        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    def timelimit(self, args):
        """
        >> timelimit <seconds>
        Sets the wall-clock time limit used by 'solve'.
        - Accepts a single non-negative integer.
        """
        if not self.arg_check(args, "s"):
            return False

        self.timelimit = int(args[0])
        return True
    
    # The following functions do not need to be callable as commands in assignment 2, but implement the PoE2 game environment for you.
    # Feel free to change or modify or replace as needed, your implementation of A1 may provide better optimized methods.
    # These functions work, but are not necessarily computationally efficient.
    # There are different approaches to exploring state spaces, this starter code provides one approach, but you are not required to use these functions.

    def get_moves(self):
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    moves.append((x, y))
        return moves

    def make_move(self, x, y):
        self.board[y][x] = self.player
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def undo_move(self, x, y):
        self.board[y][x] = 0
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    # Returns p1_score, p2_score
    def calculate_score(self):
        p1_score = 0
        p2_score = self.handicap

        # Progress from left-to-right, top-to-bottom
        # We define lines to start at the topmost (and for horizontal lines leftmost) point of that line
        # At each point, score the lines which start at that point
        # By only scoring the starting points of lines, we never score line subsets
        for y in range(self.height):
            for x in range(self.width):
                c = self.board[y][x]
                if c != 0:
                    lone_piece = True # Keep track of the special case of a lone piece
                    # Horizontal
                    hl = 1
                    if x == 0 or self.board[y][x-1] != c: #Check if this is the start of a horizontal line
                        x1 = x+1
                        while x1 < self.width and self.board[y][x1] == c: #Count to the end
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y-1][x] != c: #Check if this is the start of a vertical line
                        y1 = y+1
                        while y1 < self.height and self.board[y1][x] == c: #Count to the end
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y-1][x-1] != c: #Check if this is the start of a diagonal line
                        x1 = x+1
                        y1 = y+1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anit-diagonal
                    al = 1
                    if y == 0 or x == self.width-1 or self.board[y-1][x+1] != c: #Check if this is the start of an anti-diagonal line
                        x1 = x-1
                        y1 = y+1
                        while x1 >= 0 and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            al += 1
                            x1 -= 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Add scores for found lines
                    for line_length in [hl, vl, dl, al]:
                        if line_length > 1:
                            if c == 1:
                                p1_score += 2 ** (line_length-1)
                            else:
                                p2_score += 2 ** (line_length-1)
                    # If all found lines are length 1, check if it is the special case of a lone piece
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1

        return p1_score, p2_score
    
    def score(self, args):
        print(self.calculate_score())
    
    # Returns is_terminal, wkey_sum
    # Assumes no draws
    def is_terminal(self):
        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff:
            return True, 1
        elif p2_score >= self.score_cutoff:
            return True, 2
        else:
            # Check if the board is full
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] == 0:
                        return False, 0
            # The board is full, assign the win to the greater scoring player
            if p1_score > p2_score:
                return True, 1
            else:
                return True, 2
    
    
    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True
            
    
    def is_pos_avail(self, c, r):
        return self.board[r][c] == 0        
    
    # this function may be modified as needed, but should behave as expected
    def play(self, args):
        '''
            >> play <col> <row>
            Places current player's piece at position (<col>, <row>).
        '''
        if not self.arg_check(args, "x y"):
            return False
        
        try:
            col = int(args[0])
            row = int(args[1])
        except ValueError:
            #print("Illegal move: " + " ".join(args), file=sys.stderr)
            return False
        
        if col < 0 or col >= len(self.board[0]) or row < 0 or row >= len(self.board) or not self.is_pos_avail(col, row):
            raise Exception("Illegal Move")
            
        
        scores = self.calculate_score()
        if scores[0] >= self.score_cutoff or scores[1] >= self.score_cutoff:
            #print("Illegal move: " + " ".join(args), "game ended.", file=sys.stderr)
            return False

        # put the piece onto the board
        self.board[row][col] = self.player

        # compute the score for both players after each round
        self.calculate_score()

        # record the move
        # self.move_history.append((col, row, self.player))

        # switch player
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

        
        return True
    
        
    
    # new function to be implemented for assignment 3
    def load_patterns(self, args):
        # Get file name
        path = args[0]
        # Get next patterns
        next_patterns = []
        # Read file in Python statement, citation: https://www.freecodecamp.org/news/with-open-in-python-with-statement-syntax-example/
        with open(path) as file:
            for line in file:
                pattern, weight = line.split()
                next_patterns.append((pattern, float(weight)))
        self.patterns = next_patterns
        return True
       

    # new function to be implemented for assignment 3
    def policy_moves(self, args):
        # Compute move values in row-major order
        move_values = []
        coords = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    self.make_move(x, y)
                    # Negate the child's value as negamax
                    value = - self.calculate_weights(self.player)
                    # Undo the evaluated move to restore the board
                    self.undo_move(x, y)
                    move_values.append(value)
                    coords.append((x, y))
        # minj(mj)
        min_value = min(move_values)
        # numerator: mi - minj(mj) + 1
        numerator = [value - min_value + 1 for value in move_values]
        # denominator: sum(numerator) in |M|
        denominator = sum(numerator)
        # round (p, 3)
        probs = [round(n / denominator, 3) for n in numerator]
        print(" ".join(str(p) for p in probs))
        return True

    # new function to be implemented for assignment 3
    def move_evaluation(self, args):
        values = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    self.make_move(x, y)
                    # Negate the child's value as negamax
                    value = - self.calculate_weights(self.player)
                    # Undo the evaluated move to restore the board
                    self.undo_move(x, y)
                    values.append(value)
        out = []
        for v in values:
            r = round(v, 1)
            # If the rounded result is 0.0, then just make it 0
            if v == 0:
                r = 0
            out.append(str(r))
        print(" ".join(out))
        return True

    # new function to be implemented for assignment 3
    def position_evaluation(self, args):
        score = self.calculate_weights(self.player)
        print(score)
        return True

    # Helper method to match given pattern on board
    def match_pattern(self, x0, y0, dx, dy, pattern, current_player):
        coords = []
        opponent = 2 if current_player == 1 else 1
        for i, character in enumerate(pattern):
            x = x0 + dx * i
            y = y0 + dy * i
            if character == 'X':
                # Return None when X should be passed the boundary but appear on the board
                if (0 <= x < self.width and 0 <= y < self.height):
                    return None
                # Do not include out-of-bounds positions
                continue
            # Return None when coordinate is out of the board
            if not (0 <= x < self.width and 0 <= y < self.height):
                return None
            
            cell = self.board[y][x]
            if character == '*':
                coords.append((x, y))
            elif character == '_':
                # Current cell should be 0
                if cell != 0:
                    return None
                coords.append((x, y))
            elif character == 'P':
                # Current cell should be played by current player
                if cell != current_player:
                    return None
                coords.append((x, y))
            elif character == 'O':
                # Current cell should be played by opponent
                if cell != opponent:
                    return None
                coords.append((x, y))
        

        fx, fy = coords[0]
        if dx == 1 and dy == 0:
            # Row identifier
            identifier = ('r', fy)
        elif dx == 0 and dy == 1:
            # Column identifier
            identifier = ('c', fx)
        elif dx == 1 and dy == 1:
            # Diagonal identifier
            identifier = ('d', fy - fx)
        elif dx == -1 and dy == 1:
            # Anti-diagonal identifier
            identifier = ('ad', fy + fx)
        return coords, identifier

    # Helper method to calculate the weights of current player
    def calculate_weights(self, player):
        # Row, Column, Diagonal, Anti-diagonal
        dirs = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        # Two dictionaries used to record types of hit applied to the line and sum of weights of the line
        # Citation: LLM is used to come up with this idea and logic of using different keys for weights
        line_hits = {} 
        line_sum = {} 

        for pattern, weight in self.patterns:
            pattern_forward = pattern
            # Citation: https://stackoverflow.com/questions/931092/how-do-i-reverse-a-string-in-python
            # Reverse the string of pattern 
            pattern_back = pattern[::-1]

            # Avoid double counting if pattern_forward matches pattern_backward
            pattern_same = []
            if pattern_forward == pattern_back:
                pattern_same.append(pattern_forward)
            else:
                pattern_same.append(pattern_forward)
                pattern_same.append(pattern_back)

            for p in pattern_same:
                # Cover for 'X' case out of the board
                y_start = -len(p) + 1
                y_end = self.height
                x_start = -len(p) + 1
                x_end = self.width
                
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        for dx, dy in dirs:
                            temp = self.match_pattern(x, y, dx, dy, p, player)
                            if temp is None:
                                continue
                            coords, identifier = temp
                            type = identifier[0]
                            # Citation: https://www.geeksforgeeks.org/python/frozenset-in-python/
                            # We use frozenset since "set" is unhashable type and frozenset can be used as keys in dictionary
                            key = frozenset(coords)
                            # Record identifier for current coords set
                            if key not in line_hits:
                                line_hits[key] = set()
                            line_hits[key].add(identifier)
                            if key not in line_sum:
                                line_sum[key] = {}
                            key_sum = line_sum[key]
                            key_match = (pattern_forward, type)
                            prev = key_sum.get(key_match)
                            # If line length > 1, sum up the same patterns
                            if len(coords) > 1:
                                if prev is None:
                                    key_sum[key_match] = weight
                                else:
                                    key_sum[key_match] = prev + weight
                            # Else keep max to avoid double-counting
                            elif len(coords) == 1:
                                if prev is None or weight > prev:
                                    key_sum[key_match] = weight

        # We first obtain the max weight of the pair (pattern, type), then we remove subsets after reordering from largest to smallest sets
        max_set = {}
        for key, pattern_value_pair in line_sum.items():
            pattern_max_value = {}
            for (pattern, type), weight in pattern_value_pair.items():
                prev = pattern_max_value.get((pattern, type))
                if prev is None or weight > prev:
                    pattern_max_value[(pattern, type)] = weight
            max_set[key] = max(pattern_max_value.values())
        items = [(key, weight) for key, weight in max_set.items()]
        # We gradually remove the overlapped and smaller subsets from large to small order
        items.sort(key=lambda item: len(item[0]), reverse=True)
        kept_sets = []
        total = 0.0
        for coordinate_set, weight in items:
            overlapped = False
            for set_keep in kept_sets:
                    # Jump those included line
                    if coordinate_set < set_keep:
                        overlapped = True
                        break
            if not overlapped:
                kept_sets.append(coordinate_set)
                total += weight
                    
        return float(total)
    
   
if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()