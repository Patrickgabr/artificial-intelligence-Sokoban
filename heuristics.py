from sokoban import Map, BOX_LEFT, BOX_RIGHT, BOX_UP, BOX_DOWN

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def min_distance_to_targets(state):
    #Calculeaza distanta minima totala intre fiecare cutie si fiecare tinta
    total_distance = 0
    
    for box_name, box in state.boxes.items():
        min_dist = float('inf')
        for target_x, target_y in state.targets:
            dist = manhattan_distance(box.x, box.y, target_x, target_y)
            min_dist = min(min_dist, dist)
        total_distance += min_dist
    
    return total_distance

def min_matching_distance(state):
    #Calculeaza distanta minima totala pe baza unei asocieri optime cutie-tinta
    if len(state.boxes) != len(state.targets):
        return min_distance_to_targets(state)
    

    #extrage pozitiile cutiilor si tintelor
    box_positions = [(box.x, box.y) for box in state.boxes.values()]
    target_positions = state.targets
    
    total_distance = 0
    remaining_targets = list(target_positions)
    
    #atribuie fiecarei cutii tinta cea mai apropiata
    for box_x, box_y in box_positions:
        best_dist = float('inf')
        best_target = None
        
        for target_x, target_y in remaining_targets:
            dist = manhattan_distance(box_x, box_y, target_x, target_y)
            if dist < best_dist:
                best_dist = dist
                best_target = (target_x, target_y)
        
        if best_target:
            total_distance += best_dist
            remaining_targets.remove(best_target)
    
    return total_distance

def box_player_distance(state):
    #calculeaza distanta minima dintre jucator si cutiile care nu sunt pe tinta
    min_dist = float('inf')
    
    boxes_not_on_targets = []
    #caut cutiile care nu sunt pe tinta
    for box_name, box in state.boxes.items():
        is_on_target = False
        for target_x, target_y in state.targets:
            if box.x == target_x and box.y == target_y:
                is_on_target = True
                break
        
        if not is_on_target:
            boxes_not_on_targets.append(box)
    
    if not boxes_not_on_targets:
        return 0
    
    for box in boxes_not_on_targets:
        dist = manhattan_distance(state.player.x, state.player.y, box.x, box.y)
        min_dist = min(min_dist, dist)
    
    return min_dist

def deadlock_detection(state):
    #detecteaza blocajele in care cutiile nu pot fi mutate
    for box_name, box in state.boxes.items():
        is_on_target = False
        for target_x, target_y in state.targets:
            if box.x == target_x and box.y == target_y:
                is_on_target = True
                break
        
        if is_on_target:
            continue
        
        x, y = box.x, box.y
        
        #verific daca cutia este intr-un colt
        positions = [
            (x-1, y),
            (x+1, y),
            (x, y-1),
            (x, y+1)
        ]
        
        blocked_horizontal = 0
        blocked_vertical = 0
        
        #verific daca sunt blocaje orizontale+verticale
        if (x, y-1) in state.obstacles or (x, y+1) in state.obstacles:
            blocked_horizontal = 1
        
        if (x-1, y) in state.obstacles or (x+1, y) in state.obstacles:
            blocked_vertical = 1
        
        if blocked_horizontal and blocked_vertical:
            return 1000
    
    return 0

def pull_move_penalty(state):
    #penalizeaza mutarile de tip pull
    penalty = 0
    
    for box_name, box in state.boxes.items():
        is_on_target = False
        for target_x, target_y in state.targets:
            if box.x == target_x and box.y == target_y:
                is_on_target = True
                break
        
        if is_on_target:
            continue
        
        x, y = box.x, box.y
        
        obstacles_count = 0
        if (x-1, y) in state.obstacles:
            obstacles_count += 1
        if (x+1, y) in state.obstacles:
            obstacles_count += 1
        if (x, y-1) in state.obstacles:
            obstacles_count += 1
        if (x, y+1) in state.obstacles:
            obstacles_count += 1
        
        penalty += obstacles_count * 0.5
    
    return penalty

def distance_to_goal_state(state):
    #calculeaza distanta fata de starea tinta
    boxes_on_target = 0
    for target_x, target_y in state.targets:
        for box_name, box in state.boxes.items():
            if box.x == target_x and box.y == target_y:
                boxes_on_target += 1
                break
    
    #calculeaz procentajul de cutii pe tinta
    completion_percentage = boxes_on_target / len(state.targets)
    
    return (1 - completion_percentage) * 10

def combined_heuristic(state):
    #combin mai multe functii de heuristica
    h1 = min_matching_distance(state) * 1.0
    h2 = box_player_distance(state) * 0.5
    h3 = deadlock_detection(state)
    h4 = pull_move_penalty(state) * 0.3
    h5 = distance_to_goal_state(state) * 2.0
    
    return h1 + h2 + h3 + h4 + h5
