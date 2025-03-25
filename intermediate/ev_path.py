import heapq
import matplotlib.pyplot as plt
import networkx as nx
import geopy.geocoders
from haversine import haversine, Unit

#=====================#
#  HELPER FUNCTIONS   #
#=====================#

def new_heuristic(node, end, graph):
    """
    Calculates the Haversine distance between two nodes as a heuristic.
    """
    lat1, lon1 = graph[node]['y'], graph[node]['x']
    lat2, lon2 = graph[end]['y'], graph[end]['x']
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)

def geocode_location(location_name):
    """
    Geocodes a location name to latitude and longitude using geopy.
    """
    geolocator = geopy.geocoders.Nominatim(user_agent="ev_path_finder")
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding error for {location_name}: {e}")
        return None, None

def calculate_charging_time(charging_power, current_battery, battery_capacity):
    """
    Calculates the time needed to charge the battery to full.
    """
    if charging_power is None or charging_power == 0:
        return 0
    charge_needed = battery_capacity - current_battery
    charging_time = charge_needed / charging_power
    return charging_time

def create_simplified_graph(start_location, end_location, charging_stations_list, charging_stations_data):
    """
    Creates a simplified, connected graph for EV pathfinding.
    Each node contains coordinates (x, y) plus the charging speed (if any).
    """
    graph = {}
    locations = [start_location, end_location] + [station["name"] for station in charging_stations_data]
    print("Locations considered for graph:", locations)

    coords = {}
    # Collect coordinates for each location, skipping failures
    for loc in locations:
        lat, lon = geocode_location(loc)
        if lat and lon:
            coords[loc] = (lat, lon)
        else:
            print(f"Error: Could not geocode {loc}. Excluding from the graph.")

    # Build the graph structure
    for loc in coords:
        lat, lon = coords[loc]
        charging_power = None
        for station in charging_stations_data:
            if station["name"] == loc:
                charging_power = station["speed_kw"]
                break
        graph[loc] = {
            'neighbors': {},
            'x': lon,
            'y': lat,
            'speed_kw': charging_power
        }

    # Compute distances between each pair of locations
    for loc1 in coords:
        for loc2 in coords:
            if loc1 != loc2:
                lat1, lon1 = coords[loc1]
                lat2, lon2 = coords[loc2]
                distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)
                graph[loc1]['neighbors'][loc2] = distance

    return graph

#========================#
#   CORE PATHFINDING     #
#========================#

def shortest_ev_path_new(graph, start, end, battery_capacity, charging_stations, charge_rate,
                         battery_drain_rate, optimize_battery_use=True, car_model="Generic EV"):
    """
    Enhanced A* search algorithm for EV routing with charging considerations.
    Returns:
        (path, distance, battery_levels, car_model, charging_details)
    or
        (None, None, None, None, None) if no route is found.
    """
    # open_set items: (f_score, distance_so_far, current_node, current_battery, path, charging_stops)
    open_set = [(0, 0, start, battery_capacity, [], [])]
    closed_set = set()
    distances = {start: 0}

    while open_set:
        _, current_distance, current_node, current_battery, path_so_far, charging_so_far = heapq.heappop(open_set)
        state_key = (current_node, int(current_battery))

        # Avoid revisiting states we've already explored with a better battery condition
        if state_key in closed_set:
            continue
        closed_set.add(state_key)

        # Check if we've reached our destination
        if current_node == end:
            final_path = path_so_far + [current_node]

            # Calculate battery levels across the final path
            battery_levels = {}
            current_batt = battery_capacity
            for i in range(len(final_path)):
                node = final_path[i]
                battery_levels[node] = current_batt
                if i < len(final_path) - 1:
                    next_node = final_path[i + 1]
                    dist = graph[node]['neighbors'][next_node]
                    current_batt -= dist * battery_drain_rate

                    # Optionally charge if next node is a station and battery is below threshold
                    if next_node in charging_stations and current_batt < 0.8 * battery_capacity:
                        current_batt = battery_capacity

            return final_path, current_distance, battery_levels, car_model, charging_so_far

        # Explore neighbors
        for neighbor, distance in graph[current_node]['neighbors'].items():
            battery_consumption = distance * battery_drain_rate
            new_battery = current_battery - battery_consumption

            if new_battery <= 0:
                continue

            new_distance = current_distance + distance
            # Update known distance if this path is better
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                new_path = path_so_far + [current_node]
                new_charging = list(charging_so_far)

                # If neighbor is a charging station and battery is below threshold, charge
                if neighbor in charging_stations and new_battery < 0.8 * battery_capacity:
                    charging_power = graph[neighbor].get('speed_kw', 100)  # default 100 kW
                    charging_time = calculate_charging_time(charging_power, new_battery, battery_capacity)
                    if charging_time > 0:
                        new_charging.append({
                            "station_name": neighbor,
                            "charging_time": charging_time
                        })
                    new_battery = battery_capacity

                # f_score = distance traveled + heuristic to end
                h_score = new_heuristic(neighbor, end, graph)
                f_score = new_distance + h_score

                heapq.heappush(
                    open_set,
                    (f_score, new_distance, neighbor, new_battery, new_path, new_charging)
                )

    # If we exhaust our open_set without reaching the end, no route was found
    return None, None, None, None, None

#=========================#
#   VISUALIZATION TOOLS   #
#=========================#

def visualize_path(graph, path, charging_stations, car_model):
    """
    Creates a directed graph using networkx, colored by start, end, path, etc.
    Displays it with matplotlib.
    """
    if path is None:
        print("No path to visualize.")
        return

    G = nx.DiGraph()

    # Add all nodes with positions
    for node in graph:
        G.add_node(node, pos=(graph[node]['x'], graph[node]['y']))

    # Add only edges from the path, for clarity
    edge_labels = {}
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        distance = graph[source]['neighbors'][target]
        G.add_edge(source, target, weight=distance)
        edge_labels[(source, target)] = f"{distance:.1f} km"

    # Node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Color code nodes
    node_colors = {}
    for node in G.nodes():
        if node == path[0]:
            node_colors[node] = 'green'       # start
        elif node == path[-1]:
            node_colors[node] = 'red'         # end
        elif node in charging_stations and node in path:
            node_colors[node] = 'orange'      # charging station in path
        elif node in path:
            node_colors[node] = 'lightblue'   # route node
        else:
            node_colors[node] = 'lightgray'   # unused node

    color_list = [node_colors.get(node, 'lightgray') for node in G.nodes()]

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=color_list)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', width=2, edge_color='blue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Attempt safe edge label drawing
    try:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    except ValueError as e:
        print(f"Edge labeling error: {e}. Skipping edge labels.")

    plt.title(f"EV Route: {path[0]} → {path[-1]} ({car_model})", fontsize=20)
    plt.axis('off')

    # Custom legend
    legend_items = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Destination'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=15, label='Charging Station'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Route')
    ]
    plt.legend(handles=legend_items, loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()

def visualize_battery_usage(path, battery_levels, graph, battery_capacity):
    """
    Plots battery level vs. distance traveled along the path.
    This helps you see how the battery depletes and charges at each stop.
    """
    if not path or not battery_levels:
        print("No battery data to visualize.")
        return

    # Gather distance traveled cumulatively and battery level
    distances = [0]
    battery_data = [battery_levels[path[0]]]

    total_distance = 0
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        dist = graph[start_node]['neighbors'][end_node]
        total_distance += dist
        distances.append(total_distance)
        battery_data.append(battery_levels[end_node])

    plt.figure(figsize=(10, 6))
    plt.plot(distances, battery_data, marker='o', linewidth=2, color='green')
    plt.title("Battery Level vs. Distance Traveled", fontsize=16)
    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("Battery Level (kWh)", fontsize=12)
    plt.ylim(0, battery_capacity + 5)  # a bit of margin above capacity
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#====================#
#    MAIN FUNCTION   #
#====================#

def main():
    # EV parameters
    start_location = "Montreal, Quebec"
    end_location = "Halifax, Nova Scotia"
    car_model = "Tesla Model 3"
    battery_capacity = 82     # kWh
    charge_rate = 70          # kWh (amount to charge if needed, not the rate per km)
    battery_drain_rate = 0.12 # kWh/km

    # Sample charging station data
    charging_stations_data = [
        {"name": "Trois-Rivieres, Quebec", "speed_kw": 150},
        {"name": "Quebec City, Quebec",   "speed_kw": 100},
        {"name": "Riviere-du-Loup, Quebec","speed_kw": 50},
        {"name": "Edmundston, New Brunswick","speed_kw": 150},
        {"name": "Fredericton, New Brunswick","speed_kw": 100},
        {"name": "Moncton, New Brunswick","speed_kw": 50},
        {"name": "Truro, Nova Scotia","speed_kw": 150},
    ]

    # Build a list of station names for quick membership checks
    charging_stations_list = [station["name"] for station in charging_stations_data]
    print("Available charging stations:", charging_stations_list)

    # Create the graph
    graph = create_simplified_graph(start_location, end_location, charging_stations_list, charging_stations_data)
    if not graph:
        print("Could not create the graph (no valid geocoding). Exiting.")
        return

    # Convert charging station list to a set for quick "in" lookups
    charging_stations = set(charging_stations_list)

    # Run the pathfinding
    path, distance, battery_levels, car_model_used, charging_details = shortest_ev_path_new(
        graph,
        start_location,
        end_location,
        battery_capacity,
        charging_stations,
        charge_rate,
        battery_drain_rate,
        car_model=car_model
    )

    if not path:
        print("No viable route found with these battery constraints.")
        print("Try increasing battery capacity or adding more charging stations.")
        return

    #=========================#
    #  PRINT ROUTE SUMMARIES  #
    #=========================#
    print("\n========== EV ROUTING SUMMARY ==========")
    print(f"Vehicle: {car_model_used}")
    print(f"Route: {' → '.join(path)}")
    print(f"Total Distance: {distance:.2f} km")

    print("\n========== ROUTE DETAILS (LEG BY LEG) ==========")
    # We'll need to reconstruct battery usage since battery_levels is assigned at the end
    # We can read from battery_levels dict to show start/end battery for each leg
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        leg_distance = graph[start_node]['neighbors'][end_node]
        start_batt = battery_levels[start_node]
        end_batt = battery_levels[end_node]
        print(f"Leg {i+1}: {start_node} → {end_node}")
        print(f"  Distance: {leg_distance:.2f} km")
        print(f"  Battery at start: {start_batt:.2f} kWh ({start_batt/battery_capacity*100:.1f}%)")
        print(f"  Battery at end:   {end_batt:.2f} kWh ({end_batt/battery_capacity*100:.1f}%)")

    #========================#
    #  PRINT CHARGING STOPS  #
    #========================#
    print("\n========== CHARGING STOPS ==========")
    if charging_details:
        total_charging_time = 0
        for station_info in charging_details:
            station_name = station_info["station_name"]
            time_charged = station_info["charging_time"]
            speed = graph[station_name].get('speed_kw', 'Unknown')
            total_charging_time += time_charged
            print(f"Charging at:    {station_name}")
            print(f"  Speed:        {speed} kW")
            print(f"  Time needed:  {time_charged:.2f} hours ({time_charged * 60:.0f} minutes)")
        print(f"Total charging time: {total_charging_time:.2f} hours ({total_charging_time*60:.0f} minutes)")
        # Estimate total trip time: drive time (assume 100km/h) + charging time
        est_trip_time = distance / 100 + total_charging_time
        print(f"Estimated total trip time: {est_trip_time:.2f} hours")
    else:
        print("No charging stops required.")

    #=====================#
    #  VISUALIZE RESULTS  #
    #=====================#
    visualize_path(graph, path, charging_stations, car_model_used)
    visualize_battery_usage(path, battery_levels, graph, battery_capacity)

if __name__ == '__main__':
    main()
