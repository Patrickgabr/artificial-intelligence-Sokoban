import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sokoban import Map
from search_methods.lrta_star import LRTAStar
from search_methods.simulated_annealing import SimulatedAnnealing
from search_methods.heuristics import (
    manhattan_distance,
    min_distance_to_targets,
    min_matching_distance,
    box_player_distance,
    deadlock_detection,
    pull_move_penalty,
    distance_to_goal_state,
    combined_heuristic
)

def test_heuristic(algorithm, heuristic_func, heuristic_name, test_map_path):
    initial_state = Map.from_yaml(test_map_path)
    
    if algorithm == 'lrta':
        solver = LRTAStar(heuristic_func, verbose=False)
    else:
        solver = SimulatedAnnealing(heuristic_func, verbose=False)
    
    start_time = time.time()
    solution = solver.solve(initial_state)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    results = {
        "heuristic": heuristic_name,
        "solved": bool(solution and solution[-1].is_solved()),
        "execution_time": execution_time,
        "states_expanded": solver.expanded_states if hasattr(solver, 'expanded_states') else 0,
        "pull_moves": solver.pull_moves_count if hasattr(solver, 'pull_moves_count') else 0,
        "path_length": len(solution) if solution else 0
    }
    
    return results

def compare_heuristics_on_all_maps(algorithm, test_maps):
    heuristics = {
        "Manhattan": min_distance_to_targets,
        "Matching": min_matching_distance,
        "Box_Player": box_player_distance,
        "Deadlock": deadlock_detection,
        "Pull_Penalty": pull_move_penalty,
        "Goal_State": distance_to_goal_state,
        "Combined": combined_heuristic
    }
    
    aggregated_results = {
        name: {
            "solved_count": 0,
            "total_states": 0,
            "total_pull_moves": 0,
            "total_path_length": 0,
            "total_time": 0,
            "num_attempts": len(test_maps)
        } for name in heuristics.keys()
    }
    
    print(f"\nTesting {algorithm.upper()} with different heuristics on all maps:")
    
    for map_path in test_maps:
        map_name = os.path.basename(map_path).split('.')[0]
        print(f"\nTesting on {map_name}:")
        
        for name, func in heuristics.items():
            print(f"  - {name}...", end="", flush=True)
            try:
                result = test_heuristic(algorithm, func, name, map_path)
                
                if result["solved"]:
                    aggregated_results[name]["solved_count"] += 1
                    aggregated_results[name]["total_states"] += result["states_expanded"]
                    aggregated_results[name]["total_pull_moves"] += result["pull_moves"]
                    aggregated_results[name]["total_path_length"] += result["path_length"]
                    aggregated_results[name]["total_time"] += result["execution_time"]
                    print(" Solved!")
                else:
                    print(" Failed to solve.")
            except Exception as e:
                print(f" Error: {e}")
    
    for name in heuristics.keys():
        solved_count = aggregated_results[name]["solved_count"]
        if solved_count > 0:
            aggregated_results[name]["avg_states"] = aggregated_results[name]["total_states"] / solved_count
            aggregated_results[name]["avg_pull_moves"] = aggregated_results[name]["total_pull_moves"] / solved_count
            aggregated_results[name]["avg_path_length"] = aggregated_results[name]["total_path_length"] / solved_count
            aggregated_results[name]["avg_time"] = aggregated_results[name]["total_time"] / solved_count
        else:
            aggregated_results[name]["avg_states"] = 0
            aggregated_results[name]["avg_pull_moves"] = 0
            aggregated_results[name]["avg_path_length"] = 0
            aggregated_results[name]["avg_time"] = 0
    
    return aggregated_results

def plot_combined_performance(algorithm, aggregated_results, save_path="heuristics_comparison"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    heuristics = list(aggregated_results.keys())
    solved_counts = [aggregated_results[h]["solved_count"] for h in heuristics]
    avg_path_lengths = [aggregated_results[h]["avg_path_length"] for h in heuristics]
    avg_states = [aggregated_results[h]["avg_states"] for h in heuristics]
    avg_pull_moves = [aggregated_results[h]["avg_pull_moves"] for h in heuristics]
    avg_times = [aggregated_results[h]["avg_time"] for h in heuristics]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{algorithm.upper()} Performance with Different Heuristics (All Maps)', fontsize=16)
    
    bars1 = ax1.bar(heuristics, avg_path_lengths, color='#1f77b4')
    ax1.set_ylabel('Average Number of Moves')
    ax1.set_title('Solution Length')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars1, solved_counts):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    bars2 = ax2.bar(heuristics, avg_states, color='#ff7f0e')
    ax2.set_ylabel('Average Count')
    ax2.set_title('States Explored')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, states in zip(bars2, avg_states):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    bars3 = ax3.bar(heuristics, avg_pull_moves, color='#2ca02c')
    ax3.set_ylabel('Average Count')
    ax3.set_title('Pull Moves')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, pulls in zip(bars3, avg_pull_moves):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    bars4 = ax4.bar(heuristics, avg_times, color='#d62728')
    ax4.set_ylabel('Average Seconds')
    ax4.set_title('Execution Time')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars4, avg_times):
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}', ha='center', va='bottom')
    
    for idx, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.text(0.02, 0.98, f'Solved maps: {[solved_counts[i] for i in range(len(heuristics))]}',
                transform=ax.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{algorithm}_combined_heuristics_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    success_rates = [count/aggregated_results[h]["num_attempts"]*100 for h, count in zip(heuristics, solved_counts)]
    bars = plt.bar(heuristics, success_rates, color='#9467bd')
    plt.ylabel('Success Rate (%)')
    plt.title(f'{algorithm.upper()} Success Rate by Heuristic')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{algorithm}_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    test_maps = [
        "tests/easy_map1.yaml",
        "tests/easy_map2.yaml",
        "tests/medium_map1.yaml",
        "tests/medium_map2.yaml",
        "tests/hard_map1.yaml",
        "tests/hard_map2.yaml",
        "tests/large_map1.yaml",
        "tests/large_map2.yaml",
        "tests/super_hard_map1.yaml"
    ]
    
    lrta_results = compare_heuristics_on_all_maps('lrta', test_maps)
    plot_combined_performance('lrta', lrta_results)
    
    sa_results = compare_heuristics_on_all_maps('simulated_annealing', test_maps)
    plot_combined_performance('simulated_annealing', sa_results)
    
    print(f"\nCombined comparison charts saved in 'heuristics_comparison' directory")

if __name__ == "__main__":
    main()
