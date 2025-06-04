from sokoban import Map, moves_meaning
from search_methods.lrta_star import LRTAStar
from search_methods.simulated_annealing import SimulatedAnnealing
from search_methods.heuristics import combined_heuristic
import os
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np

def run_single_test(algorithm, map_path):
    map_name = os.path.basename(map_path).split('.')[0]
    initial_state = Map.from_yaml(map_path)
    
    print(f"Running {algorithm} on {map_name}...")
    
    if algorithm == 'lrta*':
        solver = LRTAStar(combined_heuristic, verbose=False)
    elif algorithm == 'simulated-annealing':
        solver = SimulatedAnnealing(combined_heuristic, verbose=False)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    start_time = time.time()
    solution = solver.solve(initial_state)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if solution and solution[-1].is_solved():
        print(f"  Solution found in {execution_time:.4f}s")
        print(f"  States expanded: {solver.expanded_states}")
        print(f"  Pull moves: {solver.pull_moves_count}")
        print(f"  Path length: {len(solution)}")
        
        if args.output:
            save_solution(map_name, algorithm, solution)
    else:
        print(f"  Solution not found in {execution_time:.4f}s")
    
    return {
        "map_name": map_name,
        "algorithm": algorithm,
        "solved": bool(solution and solution[-1].is_solved()),
        "execution_time": execution_time,
        "states_expanded": solver.expanded_states if hasattr(solver, 'expanded_states') else 0,
        "pull_moves": solver.pull_moves_count if hasattr(solver, 'pull_moves_count') else 0,
        "path_length": len(solution) if solution else 0
    }

def run_comparison(test_maps):
    print("Running comparison between LRTA* and Simulated Annealing...")
    
    print("\n=== Running LRTA* ===")
    lrta_results = []
    for map_path in test_maps:
        result = run_single_test('lrta*', map_path)
        lrta_results.append(result)
    
    print("\n=== Running Simulated Annealing ===")
    sa_results = []
    for map_path in test_maps:
        result = run_single_test('simulated-annealing', map_path)
        sa_results.append(result)
    
    all_results = lrta_results + sa_results
    print_summary_table(all_results)
    
    create_comparison_charts(lrta_results, sa_results)
    create_solved_chart(lrta_results, sa_results)
    save_results_csv(all_results)

def save_solution(map_name, algorithm, solution):
    output_dir = f"images/{algorithm}_{map_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, state in enumerate(solution):
        if hasattr(state, 'save_map'):
            state.save_map(output_dir, f"step_{i:03d}.png")
    
    print(f"  Solution saved in {output_dir}")

def print_summary_table(results):
    print("\n===== SUMMARY =====")
    print("=" * 90)
    print(f"{'Map':<15} {'Algorithm':<15} {'States':<10} {'Pull':<10} {'Moves':<10} {'Time (s)':<10} {'Solved?':<10}")
    print("=" * 90)
    
    for result in results:
        solved_text = "True" if result['solved'] else "False"
        print(f"{result['map_name']:<15} {result['algorithm']:<15} {result['states_expanded']:<10} "
              f"{result['pull_moves']:<10} {result['path_length']:<10} "
              f"{result['execution_time']:.4f}     {solved_text}")
    
    print("=" * 90)

def create_comparison_charts(lrta_results, sa_results):
    maps = sorted(set([r["map_name"] for r in lrta_results]))
    
    lrta_map = {r["map_name"]: r for r in lrta_results}
    sa_map = {r["map_name"]: r for r in sa_results}
    
    metrics = ["states_expanded", "pull_moves", "path_length", "execution_time"]
    metric_titles = ["Expanded states", "Pull moves", "Path length", "Execution time (s)"]
    
    plt.figure(figsize=(15, 12))
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        plt.subplot(2, 2, i+1)
        
        lrta_data = []
        sa_data = []
        
        for map_name in maps:
            if map_name in lrta_map and lrta_map[map_name]["solved"]:
                lrta_data.append(lrta_map[map_name][metric])
            else:
                lrta_data.append(0)
                
            if map_name in sa_map and sa_map[map_name]["solved"]:
                sa_data.append(sa_map[map_name][metric])
            else:
                sa_data.append(0)
        
        x = np.arange(len(maps))
        width = 0.35
        
        plt.bar(x - width/2, lrta_data, width, label='LRTA*', color='blue')
        plt.bar(x + width/2, sa_data, width, label='Simulated Annealing', color='orange')
        
        plt.xlabel('Map')
        plt.ylabel(title)
        plt.title(f'Comparison of {title}')
        plt.xticks(x, maps, rotation=45, ha='right')
        plt.legend()
        
        for j, v in enumerate(lrta_data):
            if v > 0:
                plt.text(j - width/2, v * 1.05, str(v), ha='center', va='bottom', fontsize=8)
                
        for j, v in enumerate(sa_data):
            if v > 0:
                plt.text(j + width/2, v * 1.05, str(v), ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
    
    plt.savefig("algorithm_comparison.png")
    print("Comparison chart saved as algorithm_comparison.png")
    plt.close()

def create_solved_chart(lrta_results, sa_results):
    maps = sorted(set([r["map_name"] for r in lrta_results]))
    
    lrta_map = {r["map_name"]: r for r in lrta_results}
    sa_map = {r["map_name"]: r for r in sa_results}
    
    lrta_solved = [1 if lrta_map[m]["solved"] else 0 for m in maps]
    sa_solved = [1 if sa_map[m]["solved"] else 0 for m in maps]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(maps))
    width = 0.35
    
    plt.bar(x - width/2, lrta_solved, width, label='LRTA*', color='blue')
    plt.bar(x + width/2, sa_solved, width, label='Simulated Annealing', color='orange')
    
    plt.xlabel('Map')
    plt.ylabel('Solved (1=Yes, 0=No)')
    plt.title('Maps solved by each algorithm')
    plt.xticks(x, maps, rotation=45, ha='right')
    plt.yticks([0, 1])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("solved_comparison.png")
    print("Solved chart saved as solved_comparison.png")
    plt.close()

def save_results_csv(all_results):
    import csv
    
    with open('algorithm_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['map_name', 'algorithm', 'solved', 'execution_time', 
                      'states_expanded', 'pull_moves', 'path_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            row = {k: result[k] for k in fieldnames}
            writer.writerow(row)
    
    print("Results saved to algorithm_results.csv")

def run_heuristic_visualization(test_maps):
    from visualize_heuristics import visualize_heuristics_for_maps
    
    print("\n=== Running heuristic visualization ===")
    print("This process may take some time...")

    results = visualize_heuristics_for_maps(test_maps)
    
    print("Heuristic visualization completed!")
    print("Check the 'heuristics' folder for:")
    print("  - Graphs of heuristic evolution for each map")
    print("  - Comparative analysis graphs")
    print("  - Heuristic trends based on difficulty")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sokoban solver using LRTA* or Simulated Annealing')
    parser.add_argument('algorithm', 
                        choices=['lrta*', 'simulated-annealing', 'comparison', 'heuristics'], 
                        help='The algorithm to use, comparison for both or heuristics for visualization')
    parser.add_argument('input', nargs='?', help='Path to the map file or "all" to test all maps')
    parser.add_argument('--output', action='store_true', help='Save solution images')
    parser.add_argument('--verbose', action='store_true', help='Show detailed steps of the solution')
    
    args = parser.parse_args()
    
    test_maps = [
        'tests/easy_map1.yaml',
        'tests/easy_map2.yaml',
        'tests/medium_map1.yaml',
        'tests/medium_map2.yaml',
        'tests/hard_map1.yaml',
        'tests/hard_map2.yaml',
        'tests/large_map1.yaml',
        'tests/large_map2.yaml',
        'tests/super_hard_map1.yaml'
    ]
    
    if args.algorithm == 'comparison':
        run_comparison(test_maps)
    elif args.algorithm == 'heuristics':
        run_heuristic_visualization(test_maps)
    else:
        if not args.input:
            parser.error("Input file is required in single algorithm mode")
        
        if args.input == 'all':
            results = []
            for map_path in test_maps:
                result = run_single_test(args.algorithm, map_path)
                results.append(result)
            
            print_summary_table(results)
            
        else:
            if not os.path.exists(args.input):
                print(f"Error: Map file {args.input} does not exist")
                exit(1)
                
            run_single_test(args.algorithm, args.input)
