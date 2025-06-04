from sokoban import Map, moves_meaning
from search_methods.solver import Solver
import random

class LRTAStar(Solver):

    def __init__(self, heuristic_function, max_iterations=5000, verbose=False):
        self.heuristic_function = heuristic_function
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.h_table = {}
        self.visited_states = {}
        self.solution_path = []
        self.expanded_states = 0
        self.pull_moves_count = 0
        
    def log(self, message):
        if self.verbose:
            print(message)
            
    def solve(self, initial_state):
        if initial_state.is_solved():
            return [initial_state]
        
        random.seed(42)
        
        current_state = initial_state
        self.solution_path = [current_state]
        iterations = 0
        
        stuck_count = 0
        last_h_value = float('inf')
        best_state_so_far = current_state
        best_state_h = float('inf')
        
        self.log("Inceperea algoritmului LRTA*...")
        
        while not current_state.is_solved() and iterations < self.max_iterations:
            state_str = str(current_state)
            
            if state_str in self.visited_states:
                self.visited_states[state_str] += 1
                if self.visited_states[state_str] > 10:
                    self.log(f"Warning: Starea {self.visited_states[state_str]} a fost vizitata de {self.visited_states[state_str]} ori")
                    stuck_count += 1
            else:
                self.visited_states[state_str] = 1
            
            neighbors = current_state.get_neighbours()
            self.expanded_states += 1
            
            if not neighbors:
                self.log("Nu exista mutari valide disponibile. Puzzle-ul ar putea fi imposibil de rezolvat.")
                break
            
            best_neighbors = []
            best_f_value = float('inf')
            
            for move in current_state.filter_possible_moves():
                next_state = current_state.copy()
                next_state.apply_move(move)
                next_state_str = str(next_state)
                
                if next_state_str in self.h_table:
                    h_value = self.h_table[next_state_str]
                else:
                    h_value = self.heuristic_function(next_state)
                    self.h_table[next_state_str] = h_value
                
                f_value = h_value
                
                if move >= 5:
                    f_value += 0.5
                
                if next_state_str in self.visited_states:
                    visit_penalty = min(5, self.visited_states[next_state_str]) * 0.2
                    f_value += visit_penalty
                
                if f_value < best_f_value:
                    best_f_value = f_value
                    best_neighbors = [(next_state, move)]
                elif f_value == best_f_value:
                    best_neighbors.append((next_state, move))
            
            if best_f_value < best_state_h:
                best_state_h = best_f_value
                best_state_so_far = best_neighbors[0][0]
            
            best_neighbor, move_used = random.choice(best_neighbors)
            
            if state_str not in self.h_table:
                self.h_table[state_str] = self.heuristic_function(current_state)
            
            new_h = max(self.h_table[state_str], 1 + best_f_value)
            self.h_table[state_str] = new_h
            
            if move_used >= 5:
                self.pull_moves_count += 1
                self.log(f"Folosim mutare de tip pull: {moves_meaning[move_used]}")
            else:
                self.log(f"Folosim mutare de tip push: {moves_meaning[move_used]}")
            
            if abs(new_h - last_h_value) < 0.001:
                stuck_count += 1
            else:
                stuck_count = 0
            
            last_h_value = new_h
            
            if stuck_count > 50:
                self.log("Blocare intr-un minim local, incercam mutare aleatoare...")
                moves = current_state.filter_possible_moves()
                if moves:
                    random_move = random.choice(moves)
                    next_state = current_state.copy()
                    next_state.apply_move(random_move)
                    best_neighbor = next_state
                    move_used = random_move
                    stuck_count = 0
            
            current_state = best_neighbor
            self.solution_path.append(current_state)
            
            iterations += 1
            
            if iterations % 100 == 0:
                self.log(f"Iteratia {iterations}: Am explorat {self.expanded_states} stari, Mutari de tip pull: {self.pull_moves_count}")
                if current_state.is_solved():
                    self.log("Solutia a fost gasita!")
                    break
        
        if current_state.is_solved():
            self.log(f"Solutia a fost gasita in {iterations} iteratii!")
            self.log(f"Numarul total de stari explorate: {self.expanded_states}")
            self.log(f"Numarul total de mutari de tip pull: {self.pull_moves_count}")
            return self.solution_path
        else:
            self.log(f"Nu am gasit solutia completa in {self.max_iterations} iteratii.")
            self.log(f"Numarul total de stari explorate: {self.expanded_states}")
            self.log(f"Numarul total de mutari de tip pull: {self.pull_moves_count}")
            return None
