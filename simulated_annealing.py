from sokoban import Map, moves_meaning
from search_methods.solver import Solver
import random
import math
import time

class SimulatedAnnealing(Solver):

    def __init__(self, heuristic_function, max_iterations=20000, initial_temperature=200.0, 
                 cooling_rate=0.998, min_temperature=0.01, verbose=False, restarts=5):
        self.heuristic_function = heuristic_function
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.verbose = verbose
        self.restarts = restarts
        
        # Statistici
        self.expanded_states = 0
        self.pull_moves_count = 0
        self.solution_path = []
        self.best_energy = float('inf')
        self.best_state = None
        
    def log(self, message):
        if self.verbose:
            print(message)
            
    def count_boxes_on_target(self, state):
        """Contorizeaza cate cutii sunt pe pozitiile tinta."""
        count = 0
        for target_x, target_y in state.targets:
            for box_name, box in state.boxes.items():
                if box.x == target_x and box.y == target_y:
                    count += 1
                    break
        return count
            
    def get_neighbor(self, state, strategy="weighted"):
        possible_moves = state.filter_possible_moves()
        
        if not possible_moves:
            return None
            
        if strategy == "weighted":
            push_moves = [m for m in possible_moves if m < 5]
            pull_moves = [m for m in possible_moves if m >= 5]
            
            if push_moves and random.random() < 0.8:
                move = random.choice(push_moves)
            elif pull_moves:
                move = random.choice(pull_moves)
            else:
                move = random.choice(possible_moves)
        else:
            move = random.choice(possible_moves)
            
        neighbor = state.copy()
        neighbor.apply_move(move)
        
        # Tin evidenta mutarilor de tip pull
        if move >= 5:
            self.pull_moves_count += 1
            
        self.expanded_states += 1
        return neighbor
        
    def acceptance_probability(self, current_energy, new_energy, temperature, 
                              current_boxes_on_target, new_boxes_on_target):
        if new_energy < current_energy:
            return 1.0
            
        if new_boxes_on_target > current_boxes_on_target:
            return 0.9
            
        # Probabilitate standard de acceptare
        if temperature < 0.0001:
            return 0.0
        return math.exp(-(new_energy - current_energy) / temperature)
        
    def solve(self, initial_state):
        if initial_state.is_solved():
            return [initial_state]
            
        random.seed(42)
        
        best_solution = None
        best_progress = 0
        
        # Incearca mai multe restarturi
        for restart in range(self.restarts):
            current_state = initial_state
            current_energy = self.heuristic_function(current_state)
            current_boxes_on_target = self.count_boxes_on_target(current_state)
            temperature = self.initial_temperature
            
            # Urmarim mutarile pentru a reconstrui drumul
            path = [current_state]
            
            # Bucle principala cu mai putine iteratii pentru fiecare restart
            for iteration in range(self.max_iterations // self.restarts):
                if current_state.is_solved():
                    self.log(f"Soluția a fost găsită la iterația {iteration}!")
                    return path
                    
                if temperature < self.min_temperature:
                    break
                    
                # Genereaza o stare vecina
                neighbor = self.get_neighbor(current_state, "weighted")
                
                if neighbor is None:
                    break
                    
                # Calculeaza energia si cutiile pe tinte
                neighbor_energy = self.heuristic_function(neighbor)
                neighbor_boxes_on_target = self.count_boxes_on_target(neighbor)
                
                # Decide daca acceptam solutia noua
                if self.acceptance_probability(
                    current_energy, neighbor_energy, temperature,
                    current_boxes_on_target, neighbor_boxes_on_target
                ) > random.random():
                    current_state = neighbor
                    current_energy = neighbor_energy
                    current_boxes_on_target = neighbor_boxes_on_target
                    path.append(current_state)
                    
                    if neighbor_boxes_on_target > best_progress or (
                        neighbor_boxes_on_target == best_progress and 
                        current_energy < self.best_energy
                    ):
                        self.best_state = neighbor
                        self.best_energy = current_energy
                        best_progress = neighbor_boxes_on_target
                        best_solution = path.copy()
                        
                        if neighbor.is_solved():
                            return path
                    
                temperature *= self.cooling_rate
                
            if current_state.is_solved():
                return path
        
        self.solution_path = best_solution if best_solution else [initial_state]
        return best_solution
