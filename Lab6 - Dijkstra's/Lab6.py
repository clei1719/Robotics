import copy
import math
import random


'''
IMPORTANT: Read through the code before beginning implementation!
Your solution should fill in the various "TODO" items within this starter code.
'''
g_CYCLE_TIME = .100


# Default parameters will create a 4x4 grid to test with
g_MAP_SIZE_X = 2. # 2m wide
g_MAP_SIZE_Y = 1.5 # 1.5m tall
g_MAP_RESOLUTION_X = 0.5 # Each col represents 50cm
g_MAP_RESOLUTION_Y = 0.375 # Each row represents 37.5cm
g_NUM_X_CELLS = int(g_MAP_SIZE_X // g_MAP_RESOLUTION_X) # Number of columns in the grid map
g_NUM_Y_CELLS = int(g_MAP_SIZE_Y // g_MAP_RESOLUTION_Y) # Number of rows in the grid map

# Map from Lab 4: values of 0 indicate free space, 1 indicates occupied space
g_WORLD_MAP = [0] * g_NUM_Y_CELLS*g_NUM_X_CELLS # Initialize graph (grid) as array

# Source and Destination (I,J) grid coordinates
g_dest_coordinates = (5,5)
g_src_coordinates = (0,0)


def create_test_map(map_array):
   # Takes an array representing a map of the world, copies it, and adds simulated obstacles
   num_cells = len(map_array)
   new_map = copy.copy(map_array)
   # Add obstacles to up to sqrt(n) vertices of the map
   for i in range(int(math.sqrt(len(map_array)))):
       random_cell = random.randint(0, num_cells)
       new_map[random_cell] = 1

   return new_map


def vertex_index_to_ij(vertex_index):
   '''
   vertex_index: unique ID of graph vertex to be converted into grid coordinates
   Returns COL, ROW coordinates in 2D grid
   '''
   global g_NUM_X_CELLS
   return vertex_index % g_NUM_X_CELLS, vertex_index // g_NUM_X_CELLS

def ij_to_vertex_index(i,j):
   '''
   i: Column of grid map
   j: Row of grid map

   returns integer 'vertex index'
   '''
   global g_NUM_X_CELLS
   return j*g_NUM_X_CELLS + i


def ij_coordinates_to_xy_coordinates(i,j):
   '''
   i: Column of grid map
   j: Row of grid map

   returns (X, Y) coordinates in meters at the center of grid cell (i,j)
   '''
   global g_MAP_RESOLUTION_X, g_MAP_RESOLUTION_Y
   return (i+0.5)*g_MAP_RESOLUTION_X, (j+0.5)*g_MAP_RESOLUTION_Y

def xy_coordinates_to_ij_coordinates(x,y):
   '''
   i: Column of grid map
   j: Row of grid map

   returns (X, Y) coordinates in meters at the center of grid cell (i,j)
   '''
   global g_MAP_RESOLUTION_X, g_MAP_RESOLUTION_Y
   return int(i // g_MAP_RESOLUTION_X), int(j // g_MAP_RESOLUTION_Y)

# **********************************
# *      Core Dijkstra Functions   *
# **********************************

def get_travel_cost(vertex_source, vertex_dest):
   # Returns the cost of moving from vertex_source (int) to vertex_dest (int)
   # INSTRUCTIONS:


   '''
       This function should return 1 if:
         vertex_source and vertex_dest are neighbors in a 4-connected grid (i.e., N,E,S,W of each other but not diagonal) and neither is occupied in g_WORLD_MAP (i.e., g_WORLD_MAP isn't 1 for either)

       This function should return 1000 if:
         vertex_source corresponds to (i,j) coordinates outside the map
         vertex_dest corresponds to (i,j) coordinates outside the map
         vertex_source and vertex_dest are not adjacent to each other (i.e., more than 1 move away from each other)
   '''

  from_x, from_y = vertex_source
  to_x, to_y = vertex_dest
  if from_x >= g_NUM_X_CELLS or from_x < 0 or to_x >= g_NUM_X_CELLS or to_x < 0:
      return 1000
  if from_y >= g_NUM_Y_CELLS or from_y < 0 or to_y >= g_NUM_Y_CELLS or to_y < 0:
      return 1000

  if abs(from_x - to_x) == 1 or abs(from_y - to_y) == 1:
      if not (g_WORLD_MAP[vertex_source] or g_WORLD_MAP[vertex_dest]):
          return 1
  return 1000




def run_dijkstra(source_vertex):
   '''
   source_vertex: vertex index to find all paths back to
   returns: 'prev' array from a completed Dijkstra's algorithm run

   Function to return an array of ints corresponding to the 'prev' variable in Dijkstra's algorithm
   The 'prev' array stores the next vertex on the best path back to source_vertex.
   Thus, the returned array prev can be treated as a lookup table:  prev[vertex_index] = next vertex index on the path back to source_vertex
   '''
   global g_NUM_X_CELLS, g_NUM_Y_CELLS
   # Array mapping vertex_index to distance of shortest path from vertex_index to source_vertex.
   dist = [0] * g_NUM_X_CELLS * g_NUM_Y_CELLS

   # Queue for identifying which vertices are up to still be explored:
   # Will contain tuples of (vertex_index, cost), sorted such that the min cost is first to be extracted (explore cheapest/most promising vertices first)
   Q_cost = []

   # Array of ints for storing the next step (vertex_index) on the shortest path back to source_vertex for each vertex in the graph
   prev = [-1] * NUM_X_CELLS*NUM_Y_CELLS

   # Insert your Dijkstra's code here. Don't forget to initialize Q_cost properly!
   visited = [False] * len(g_WORLD_MAP)
   dist = [math.inf] * len(g_WORLD_MAP)  # unknown distance to v
   prev = [None] * len(g_WORLD_MAP)  # predecessor to v

   #don’t really need cost, could just be Q
   for v in g_WORLD_MAP:
	Q_cost.append((v,-1))
   #actually if we sort, could replace the if statement when checking for neighbors
   #Q_cost.sort(key=lambda x: x[1])
   dist[source_vertex] = 0

   while Q_cost:  # while the Q is not empty
       # vertex with least dist value, this will start with source_vertex with dist of 0
       u = dist.index(min(dist))
       # remove u
       Q_cost.pop(u)

       # Check each vertex left in Q_cost to get the neighbors of u
       for (v,cost) in Q_cost: #no need to use cost...
           #unoccupied neighbor and in bounds, might need to change to check all neighbors to get costs of 1000
           if(get_travel_cost(vertex_index_to_ij(u), vertex_index_to_ij(v)) == 1):
               alt = dist[u] + get_travel_cost(vertex_index_to_ij(u), vertex_index_to_ij(v))
               if alt < dist[v]:
                   dist[v] = alt
                   prev[v] = u

   # Return results of algorithm run
   return prev


def reconstruct_path(prev, source_vertex, dest_vertex):
   '''
   Given a populated 'prev' array, a source vertex_index, and destination vertex_index,
   allocate and return an integer array populated with the path from source to destination.
   The first entry of your path should be source_vertex and the last entry should be the dest_vertex.
   If there is no path between source_vertex and dest_vertex, as indicated by hitting a '-1' on the
   path from dest to source, return an empty list.
   '''
   final_path = []

   current = dest_vertex
   while current != -1 or current == source_vertex:
        final_path.append(current)
        current = prev[current]

   return final_path


def render_map(map_array):
   '''
     Example:
     For g_WORLD_MAP = [0, 0, 1, 0,
                        0, 1, 1, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0]
     There are obstacles at (I,J) coordinates: [ (2,0), (1,1), (2,1) ]
     The map should render as:
       .  .  .  .
       .  .  .  .
       . [ ][ ] .
       .  . [ ] .


     Make sure to display your map so that I,J coordinate (0,0) is in the bottom left.
     (To do this, you'll probably want to iterate from row 'J-1' to '0')
   '''

   # TODO: This probably has to be reversed in some way
   print(''.join([('[ ]' if space else ' . ') + (' ' if (index + 1) % math.sqrt(len(map_array)) else '\n') for index, space in enumerate(map_array)]), end='')



def main():
   global g_WORLD_MAP

   # TODO: Initialize a grid map to use for your test -- you may use create_test_map for this, or manually set one up with obstacles
   g_WORLD_MAP = create_test_map(g_WORLD_MAP)



   # Use render_map to render your initialized obstacle map

   # TODO: Find a path from the (I,J) coordinate pair in g_src_coordinates to the one in g_dest_coordinates using run_dijkstra and reconstruct_path

   '''
   TODO-
     Display the final path in the following format:
     Source: (0,0)
     Goal: (3,1)
     0 -> 1 -> 2 -> 6 -> 7
   '''


if __name__ == "__main__":
   main()
