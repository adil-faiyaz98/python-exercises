import heapq
import matplotlib.pyplot as plt
import networkx as nx
import geopy.geocoders
from haversine import haversine, Unit
import time


# Heuristic function (Haversine distance)
def new_heuristic(node, end, graph):
    """
    Calculates the Haversine distance between two nodes as a heuristic.
    """
    lat1, lon1 = graph[node]['y'], graph[node]['x']
    lat2, lon2 = graph[end]['y'], graph[end]['x']
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)


# Sample charging station data (replace with your actual data)
charging_stations_data = [
    {"name": "Trois-Rivieres, Quebec", "speed_kw": 150},
    {"name": "Quebec City, Quebec", "speed_kw": 100},
    {"name": "Riviere-du-Loup, Quebec", "speed_kw": 50},
    {"name": "Edmundston, New Brunswick", "speed_kw": 150},
    {"name": "Fredericton, New Brunswick", "speed_kw": 100},
    {"name": "Moncton, New Brunswick", "speed_kw": 50},
    {"name": "Truro, Nova Scotia", "speed_kw": 150},
]


# Fixed function to create a proper graph with correct neighbor connections

def create_simplified_graph(start_location, end_location, charging_stations_list, charging_stations_data):
    """
    Creates a simplified, connected graph for EV pathfinding.
    """
    graph = {}

    locations = [start_location, end_location] + [station["name"] for station in charging_stations_data]
    print(locations)

    # First collect all valid coordinates
    coords = {}
    for loc in locations:
        lat, lon = geocode_location(loc)
        if lat and lon:
            coords[loc] = (lat, lon)
        else:
            print(f"Error: Could not geocode {loc}. Excluding {loc} from the graph.")
            continue

    for loc in coords:
        lat, lon = coords[loc]
        # Find charging power if available
        charging_power = None
        for station in charging_stations_data:
            if station["name"] == loc:
                charging_power = station["speed_kw"]
                break
        graph[loc] = {'neighbors': {}, 'x': lon, 'y': lat, "speed_kw": charging_power}

    for loc1 in coords:
        # Initialize neighbors dictionary once (not for each connection)
        for loc2 in coords:
            if loc1 != loc2:
                lat1, lon1 = coords[loc1]
                lat2, lon2 = coords[loc2]
                distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)
                graph[loc1]['neighbors'][loc2] = distance

    print(graph)
    return graph


# Function to geocode a location (using geopy)
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


def shortest_ev_path_new(graph, start, end, battery_capacity, charging_stations, charge_rate, battery_drain_rate,
                         optimize_battery_use=True, car_model="Generic EV"):
    """
    Enhanced A* search algorithm for EV routing with charging considerations
    """
    # Priority queue: (f_score, g_score, node, battery, path, charging_details)
    open_set = [(0, 0, start, battery_capacity, [], [])]
    closed_set = set()  # Track visited states
    distances = {start: 0}  # g_score: Distance from start

    while open_set:
        _, current_distance, current_node, current_battery, path_so_far, charging_so_far = heapq.heappop(open_set)

        state_key = (current_node, int(current_battery))
        if state_key in closed_set:
            continue
        closed_set.add(state_key)

        # If we've reached the destination, return the path
        if current_node == end:
            final_path = path_so_far + [current_node]
            # Calculate battery levels along the path
            battery_levels = {}
            current_batt = battery_capacity
            for i in range(len(final_path)):
                node = final_path[i]
                battery_levels[node] = current_batt

                if i < len(final_path) - 1:
                    next_node = final_path[i + 1]
                    distance = graph[node]['neighbors'][next_node]
                    current_batt -= distance * battery_drain_rate

                    # Apply charging at stations
                    if next_node in charging_stations and current_batt < 0.8 * battery_capacity:
                        current_batt = battery_capacity  # Full charge for simplicity

            return final_path, current_distance, battery_levels, car_model, charging_so_far

        # Explore neighbors
        for neighbor, distance in graph[current_node]['neighbors'].items():
            # Calculate battery consumption
            battery_consumption = distance * battery_drain_rate
            new_battery = current_battery - battery_consumption

            # Skip if battery would be depleted
            if new_battery <= 0:
                continue

            new_distance = current_distance + distance

            # Only consider this path if it's better than any previous path to this neighbor
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance

                # Path and charging information
                new_path = path_so_far + [current_node]
                new_charging = list(charging_so_far)

                # If this is a charging station and we need charging
                if neighbor in charging_stations and new_battery < 0.8 * battery_capacity:
                    charging_power = graph[neighbor].get('speed_kw', 100)  # Default to 100kW
                    charging_time = calculate_charging_time(
                        charging_power,
                        new_battery,
                        battery_capacity
                    )

                    if charging_time > 0:
                        new_charging.append({
                            "station_name": neighbor,
                            "charging_time": charging_time
                        })

                    # Charge the battery
                    new_battery = battery_capacity

                # Calculate f_score = g_score + heuristic
                h_score = new_heuristic(neighbor, end, graph)
                f_score = new_distance + h_score

                # Add to priority queue
                heapq.heappush(open_set, (f_score, new_distance, neighbor, new_battery, new_path, new_charging))

                # No path found
    return None, None, None, None, None


# Function to calculate charging time
def calculate_charging_time(charging_power, current_battery, battery_capacity):
    """
    Calculates the time needed to charge the battery to full.
    """
    if charging_power is None or charging_power == 0:
        return 0

    charge_needed = battery_capacity - current_battery
    charging_time = charge_needed / charging_power
    return charging_time


def visualize_path(graph, path, charging_stations, car_model):
    """
    Enhanced visualization with better error handling
    """
    if path is None:
        print("No path to visualize.")
        return

    # Create directed graph for the path
    G = nx.DiGraph()

    # Add all nodes first
    for node in graph:
        G.add_node(node, pos=(graph[node]['x'], graph[node]['y']))

    # Add only the edges in the path to make visualization cleaner
    edge_labels = {}
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        distance = graph[source]['neighbors'][target]
        G.add_edge(source, target, weight=distance)
        edge_labels[(source, target)] = f"{distance:.1f}km"

    # Node positions for drawing
    pos = nx.get_node_attributes(G, 'pos')

    # Set node colors
    node_colors = {}
    for node in G.nodes():
        if node == path[0]:
            node_colors[node] = 'green'  # Start
        elif node == path[-1]:
            node_colors[node] = 'red'  # End
        elif node in charging_stations and node in path:
            node_colors[node] = 'orange'  # Charging station in path
        elif node in path:
            node_colors[node] = 'lightblue'  # Regular node in path
        else:
            node_colors[node] = 'lightgray'  # Node not in path

    # Color list in node order
    node_color_list = [node_colors.get(node, 'lightgray') for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=800)
    nx.draw_networkx_edges(G, pos, width=3, edge_color='blue', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Draw edge labels safely
    try:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    except ValueError as e:
        print(f"Visualization Error: {e}")
        print("Skipping edge labels due to layout constraints.")

    # Add title and legend
    plt.title(f"EV Route: {path[0]} → {path[-1]} ({car_model})", fontsize=18)

    # Add legend
    legend_items = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Destination'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=15, label='Charging Station'),
        plt.Line2D([0], [0], color='blue', lw=3, label='Route')
    ]
    plt.legend(handles=legend_items, loc='upper right', fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Main function with charging details display
def main():
    # EV parameters
    start_location = "Montreal, Quebec"
    end_location = "Halifax, Nova Scotia"
    car_model = "Tesla Model 3"
    battery_capacity = 82  # kWh
    charge_rate = 70  # kWh (amount to charge, not rate per km)
    battery_drain_rate = 0.12  # kWh/km

    charging_stations_list = [station["name"] for station in charging_stations_data]

    print(f"Available charging stations: {charging_stations_list}")

    # Create the graph
    graph = create_simplified_graph(start_location, end_location, charging_stations_list, charging_stations_data)
    if not graph:
        print("Could not create graph. Exiting.")
        return

    charging_stations = set(charging_stations_list)

    # Find the optimal EV route
    result = shortest_ev_path_new(
        graph, start_location, end_location, battery_capacity,
        charging_stations, charge_rate, battery_drain_rate, car_model=car_model
    )

    if result:
        path, distance, battery_levels, car_model_used, charging_details = result

        # Print route summary
        print("\n========== EV ROUTING SUMMARY ==========")
        print(f"Vehicle: {car_model_used}")
        print(f"Route: {' → '.join(path)}")
        print(f"Total Distance: {distance:.2f} km")

        # Print each leg with battery information
        print("\n========== ROUTE DETAILS (LEG BY LEG) ==========")
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            leg_distance = graph[start_node]['neighbors'][end_node]
            start_battery = battery_levels[start_node]
            end_battery = battery_levels[end_node]
            battery_used = start_battery - end_battery

            print(f"Leg {i + 1}: {start_node} → {end_node}")
            print(f"  Distance: {leg_distance:.2f} km")
            print(f"  Battery at start: {start_battery:.2f} kWh ({start_battery / battery_capacity * 100:.1f}%)")
            print(f"  Battery at end: {end_battery:.2f} kWh ({end_battery / battery_capacity * 100:.1f}%)")
            print(f"  Battery used: {battery_used:.2f} kWh")

        # Print charging stops
        print("========== CHARGING STOPS ==========")
        if charging_details:
            total_charging_time = 0
            for station in charging_details:
                if station and "station_name" in station:
                    charging_station = station["station_name"]
                    charging_time = station["charging_time"]
                    total_charging_time += charging_time
                    charging_speed = graph[charging_station].get('speed_kw', 'Unknown')

                    print(f"Charging at: {charging_station}")
                    print(f"  Charging speed: {charging_speed} kW")
                    print(f"  Charging time: {charging_time:.2f} hours ({charging_time * 60:.0f} minutes)")

            print(f"Total charging time: {total_charging_time:.2f} hours ({total_charging_time * 60:.0f} minutes)")
            print(
                f"Estimated total trip time: {distance / 100 + total_charging_time:.2f} hours")  # Assuming 100 km/h average speed
        else:
            print("No charging stops required.")

        # Visualize the route
        visualize_path(graph, path, charging_stations, car_model_used)
    else:
        print("No viable route found with current battery constraints.")
        print("Try increasing battery capacity, adding more charging stations, or choosing locations closer together.")


if __name__ == '__main__':
    main()  # Call the main function
    
    





