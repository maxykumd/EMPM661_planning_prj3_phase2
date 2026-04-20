#!/usr/bin/env python3

import numpy as np
import math
import cv2
import heapq
import json
import os

MAP_W_CM = 800
MAP_H_CM = 400
SCALE = 2 # just for visual clarity
CANVAS_W = int(MAP_W_CM * SCALE)
CANVAS_H = int(MAP_H_CM * SCALE)
ROBOT_RADIUS_CM = 22.0 # TurtleBot3 Waffle (from datasheet)
CURRENT_CLEARANCE = 0.0
GOAL_THRES = 10.0 #cm

def to_opencv_coord(x_mm, y_mm):
    col = int(round(x_mm * SCALE)) 
    row = int(round((MAP_H_CM - y_mm) * SCALE))
    return (col, row)

def color_map(obs_raw, obs_inflated):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:, :] = (50, 50, 50)
    inflate_only = (obs_inflated == 255) & (obs_raw == 0)
    canvas[inflate_only]   = (100, 0, 150)
    canvas[obs_raw == 255] = (30,  30, 220)
    for x_cm in range(0, int(MAP_W_CM)+1, 50):
        col = int(x_cm * SCALE)
        cv2.line(canvas, (col,0), (col,CANVAS_H), (80,80,80), 1)
        cv2.putText(canvas, str(x_cm), (col+2, CANVAS_H-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130,130,130), 1)
    for y_cm in range(0, int(MAP_H_CM)+1, 50):
        row = int((MAP_H_CM - y_cm) * SCALE)
        cv2.line(canvas, (0,row), (CANVAS_W,row), (80,80,80), 1)
        cv2.putText(canvas, str(y_cm), (3, row-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130,130,130), 1)
    return canvas

def is_in_obstacle(x, y, clearance_cm):
    """
    Returns True if (x,y) in cm is inside obstacle or clearance zone.
    """
    c = ROBOT_RADIUS_CM + clearance_cm #total inflation)

    # border of map ────────────────────────────────────────────────────────────────
    if x >= MAP_W_CM - 5 - c: # right wall
        return True
    if y <= 5 + c: # bottom wall
        return True
    if y >= MAP_H_CM - 5 - c: # top wall
        return True

    # left wall gate ntrance (two segments, 80cm each, gap in middle ────────────────────────────────────────────────────────────────
    # top-left segment: y=120 to y=200
    if (x <= 5 + c) and (y >= 260 - c):
        return True
    # bottom-left segment: y=0 to y=80
    if (x <= 5 + c) and (y <= 140 + c):
        return True

    # the 3 rectangle in map ────────────────────────────────────────────────────────────────
    # rect 1: center (42, 45), 30.40x30.40cm
    if (x >= 84 - 30.4 - c) and (x <= 84 + 30.4 + c) and (y >= 90 - 30.4 - c) and (y <= 90 + 30.4 + c):
        return True

    # rect 2: center (133.5, 155)
    if (x >= 267 - 30.4 - c) and (x <= 267 + 30.4 + c) and (y >= 310 - 30.4 - c) and (y <= 310 + 30.4 + c):        
        return True

    # rect 3: center (220, 174)
    if (x >= 440 - 30.4 - c) and (x <= 440 + 30.4 + c) and (y >= 352 - 30.4 - c) and (y <= 352 + 30.4 + c):
        return True

    # 2 diagonal and 1 straight wall ────────────────────────────────────────────────────────────────
    # vertical wall: x=281, y=55→200, thickness=5cm 
    if (x >= 567 - c) and (x <= 577 + c) and (y >= 110 - c):
        return True

    half_t = 5.0 + c #inflate wall by 2.5 each side to total 5thickness

    # diagonal wall 1: (38.4,200) → (108.4,78.76), 30deg 
    #14c3 = 121 14s3=70
    w1x1, w1y1 = 76.8,  400.0
    w1x2, w1y2 = 216.8, 157.52
    w1dx = w1x2 - w1x1 #how far x-dir AtoB
    w1dy = w1y2 - w1y1 #how far y-dir AtoB
    w1len = math.sqrt(w1dx**2 + w1dy**2) #wall length
    # rotate 90deg by change x,y to -y,x
    w1nx  = -w1dy / w1len #perpendic x component
    w1ny  =  w1dx / w1len
    px, py = x - w1x1, y - w1y1 
    along  = (px*w1dx + py*w1dy) / w1len #dist point from AtoB
    perp   =  px*w1nx + py*w1ny #how far point is to side of wall , p-0=wall centerline 
    extend = 20
    if (-half_t <= perp <= half_t) and (-c -extend <= along <= w1len+c+extend):
        return True

    # circle caps at each tip
    if math.sqrt((x-w1x1)**2 + (y-w1y1)**2) <= half_t:   # tip A cap
        return True
    if math.sqrt((x-w1x2)**2 + (y-w1y2)**2) <= half_t:   # tip B cap
        return True

    # diagonal wall 2: (126,5) → (172.17,131.86), 20deg 
    w2x1, w2y1 = 252.0, 10.0
    w2x2, w2y2 = 387.0, 243.82
    w2dx = w2x2 - w2x1
    w2dy = w2y2 - w2y1
    w2len = math.sqrt(w2dx**2 + w2dy**2)
    w2nx  = -w2dy / w2len
    w2ny  =  w2dx / w2len
    px, py = x - w2x1, y - w2y1
    along  = (px*w2dx + py*w2dy) / w2len
    perp   =  px*w2nx + py*w2ny
    if (-half_t <= perp <= half_t) and (-c <= along <= w2len+c):
        return True

    return False

def build_map(clearance_cm):
    """Build visual arrays using half-plane equations."""
    print("  Building obstacle map please wait ...")
    obs_raw      = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8) #create blank canvas
    obs_inflated = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    for row in range(CANVAS_H): #loop through every pixel on screen and color obstacle
        for col in range(CANVAS_W):
            x_cm = col / SCALE # convert px to cm
            y_cm = MAP_H_CM - (row / SCALE)
            if is_in_obstacle(x_cm, y_cm, 0):
                obs_raw[row, col] = 255
            if is_in_obstacle(x_cm, y_cm, clearance_cm):
                obs_inflated[row, col] = 255

    # save map for verification
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(script_dir, "map_output.png")
    canvas = color_map(obs_raw, obs_inflated)
    cv2.imwrite(map_path, canvas)
    print(f"  Map saved → {map_path}")

    return obs_raw, obs_inflated


def is_free_space(obs_inflated, x_cm, y_cm):
    """Collision check using half-plane equations directly."""
    # in map boundary ?
    if x_cm < 0 or x_cm >= MAP_W_CM or y_cm < 0 or y_cm >= MAP_H_CM:
        return False
    return not is_in_obstacle(x_cm, y_cm, CURRENT_CLEARANCE)


WHEEL_RADIUS_CM = 3.3
WHEEL_DIST_CM = 28.7
TIME_STEP_S = 0.1 # 100ms per step
TIME_PER_ACTION_S = 1.0 # 1s per action (10 steps)

def rpm_to_rads(rpm):
    return rpm * 2 * math.pi / 60.0

def differentail_drive(x, y, theta_deg, rpm_left, rpm_right, obs_inflated):
    ul = rpm_to_rads(rpm_left)
    ur = rpm_to_rads(rpm_right)
    theta_rad = math.radians(theta_deg)
    r = WHEEL_RADIUS_CM
    L = WHEEL_DIST_CM
    curr_x , curr_y = x, y
    curve_points = [(curr_x, curr_y)]
    cost = 0.0
    
    steps = int(TIME_PER_ACTION_S / TIME_STEP_S) # 10 steps for 1 action
    
    for i in range(steps):
        # calculate how much robot move,turn in each small step
        dx = (r/2) * (ul + ur) * math.cos(theta_rad) * TIME_STEP_S
        dy = (r/2) * (ul + ur) * math.sin(theta_rad) * TIME_STEP_S
        dtheta = (r/L) * (ur - ul) * TIME_STEP_S
        new_x = curr_x + dx # move forward by dx
        new_y = curr_y + dy 
        theta_rad += dtheta 
        
        if not is_free_space(obs_inflated, new_x, new_y):
            return None # action is invalid if have collision, astar skip this
        cost += math.sqrt(dx**2 + dy**2) # accumulate cost as dist travel
        curr_x, curr_y = new_x, new_y
        curve_points.append((curr_x, curr_y))
        
    new_theta_deg = math.degrees(theta_rad) % 360
    return curr_x, curr_y, new_theta_deg, cost, curve_points
    
def get_action_space(rpm1, rpm2):
    return [
        (0,rpm1),  #left_wheel_rpm, right_wheel_rpm
        (rpm1,0),    
        (rpm1,rpm1), 
        (0,rpm2), 
        (rpm2,0),    
        (rpm2,rpm2), 
        (rpm1,rpm2), 
        (rpm2,rpm1), 
    ]
    
# duplicate node detection - same cell if within the threshold
def snap_state_to_grid(x,y, theta_deg):
    XY_POS_THRES = 3 #cm
    THETA_THRES = 30 #deg
    snap_x_grid = round(x / XY_POS_THRES)
    snap_y_grid = round(y / XY_POS_THRES)
    snap_theta = round(theta_deg / THETA_THRES)
    snap_theta = snap_theta % (360 // THETA_THRES) #snap every 30deg(1-12 slot)
    return (snap_x_grid, snap_y_grid, snap_theta)
        
def euclidean_heuristic(x,y, goal_x, goal_y):
    return math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
       

def is_goal_reached(x,y, goal_x, goal_y):
    return euclidean_heuristic(x,y, goal_x, goal_y) <= GOAL_THRES

def astar(start, goal, rpm1, rpm2, obs_inflated):
    node_to_action = {snap_state_to_grid(*start): (rpm1, rpm1)}  # dummy for start
    sx, sy, st = start
    gx, gy = goal
    actions = get_action_space(rpm1, rpm2)
    tie = 0 #prevent node crash if identical f or g cost
    open_list = [] #sort by f-cost
    visited = set()
    all_curves = [] #curve draw for visualize
    cost_come = {snap_state_to_grid(*start) : 0} #g-cost per node
    parent_map = {snap_state_to_grid(*start) : None} #parent key
    parent_node = {snap_state_to_grid(*start) : start} #actual coord per node
    heapq.heappush(open_list, (euclidean_heuristic(sx,sy,gx,gy), 0, tie, (sx,sy,st))) #f=heu, g=0 at start


    while open_list:
        f, g, _, curr_state = heapq.heappop(open_list)
        node = snap_state_to_grid(*curr_state)
    
        # skip if already visited
        if node in visited:
            continue
        
        # add node to visited
        visited.add(node)
        
        # goal check
        cx, cy, ct = curr_state
        if is_goal_reached(cx, cy, gx, gy):
            print("Goal Reached")
            
            # start backtrack
            path = []
            actions_used = []  
            while node is not None:
                path.append(parent_node[node])
                if parent_map[node] is not None: #skip start node
                    actions_used.append(node_to_action[node])
                node = parent_map[node]
            path.reverse()
            actions_used.reverse()
            return path, all_curves, actions_used

        # expand all actions space
        for rpm_left, rpm_right in actions:
            result = differentail_drive(cx,cy,ct, rpm_left, rpm_right, obs_inflated)
            if result is None:
                continue
            
            # get new neighbor position
            nx, ny, nt, step_cost, curve = result
            neighbor = snap_state_to_grid(nx, ny, nt)
            if neighbor in visited:
                continue
            
            
            new_cost = g + step_cost # cost to reach this neighbor from curr node
            # update cost if see first time or cheaper path
            # store action when updating neighbor
            if neighbor not in cost_come or new_cost < cost_come[neighbor]:
                cost_come[neighbor] = new_cost #g-cost
                parent_map[neighbor] = node
                parent_node[neighbor] = (nx,ny,nt)
                node_to_action[neighbor] = (rpm_left, rpm_right)
                f_cost = new_cost + euclidean_heuristic(nx,ny,gx,gy)  
                tie +=1
                heapq.heappush(open_list, (f_cost, new_cost, tie, (nx, ny, nt)))
                all_curves.append(curve)
                
    print("No path found.")
    return None, all_curves, []
                

def visualize(obs_raw, obs_inflated, start, goal, all_curves, path, rpm1, rpm2):
    canvas = color_map(obs_raw, obs_inflated)

    print("drawing exploration tree...")
    for curve in all_curves:
        for i in range(len(curve) - 1): # connecting consecutive points to create curve
            x0, y0 = curve[i][0],   curve[i][1]
            x1, y1 = curve[i+1][0], curve[i+1][1]
            cv2.line(canvas, to_opencv_coord(x0,y0),
                     to_opencv_coord(x1,y1), (0,200,0), 1)

    if path and len(path) > 1:
        print("drawing optimal path...")
        for i in range(len(path) - 1):
            x0, y0 = path[i][0],   path[i][1]
            x1, y1 = path[i+1][0], path[i+1][1]
            cv2.line(canvas, to_opencv_coord(x0,y0),
                     to_opencv_coord(x1,y1), (255,80,80), 3)

    sp = to_opencv_coord(start[0], start[1])
    gp = to_opencv_coord(goal[0],  goal[1])
    cv2.circle(canvas, sp, 6, (0,165,255), -1)
    cv2.putText(canvas, "START", (sp[0]+8, sp[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)
    cv2.circle(canvas, gp, 6, (0,255,200), -1)
    cv2.circle(canvas, gp, int(GOAL_THRES * SCALE), (0,255,200), 1)
    cv2.putText(canvas, "GOAL", (gp[0]+8, gp[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,200), 1)
    cv2.putText(canvas, f"RPM1={rpm1} RPM2={rpm2}", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    # save to script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(script_dir, "astar_result.png")
    cv2.imwrite(result_path, canvas)
    print(f"Saved → {result_path}")
    cv2.imshow("A* Result — press any key to close", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_user_inputs(obs_inflated):
    print(f" Map: {MAP_W_CM} x {MAP_H_CM} cm | origin: bottom-left")

    while True:
        try:
            xs = float(input("Start x (cm): "))
            ys = float(input("Start y (cm): "))
            ts = float(input("Start Theta (deg, multiple of 30): "))
            if ts % 30 != 0:
                print(" Theta must be multiple of 30.\n"); continue
            if not is_free_space(obs_inflated, xs, ys):
                print(" Start is in obstacle space.\n"); continue
            break
        except ValueError:
            print(" Numbers only.\n")

    while True:
        try:
            xg = float(input("Goal x (cm): "))
            yg = float(input("Goal y (cm): "))
            if not is_free_space(obs_inflated, xg, yg):
                print(" goal is in obstacle space.\n"); continue
            break
        except ValueError:
            print(" number only.\n")

    while True:
        try:
            rpm1 = float(input("RPM1: "))
            rpm2 = float(input("RPM2: "))
            if rpm1 <= 0 or rpm2 <= 0:
                print(" rpms must be positive.\n"); continue
            break
        except ValueError:
            print(" number only.\n")

    return (xs, ys, ts % 360), (xg, yg), rpm1, rpm2


def main():
    global CURRENT_CLEARANCE

    while True: #step1 : get clearance 
        try:
            clearance_mm = float(input("Clearance (mm): "))
            clearance = float(clearance_mm / 10.0)
            if clearance < 0:
                print("  Must be >= 0.\n"); continue
            break
        except ValueError:
            print(" number only.\n")

    CURRENT_CLEARANCE = clearance

    #step2 : build obstacle map 
    print("\nDrawing obstacle map...")
    obs_raw, obs_inflated = build_map(clearance)
    #step3 : get user input
    start, goal, rpm1, rpm2 = get_user_inputs(obs_inflated)
    #step4: run Astar
    print("\nRunning A*, its thinking pls wait...")
    path, all_curves, actions_used = astar(start, goal, rpm1, rpm2, obs_inflated)
    #export path to JSON for use in the ros part
    if path is None:
        print("No path found.")
    else:
        print(f"Path found: {len(path)} nodes.")
        data = {
            "rpm1": rpm1, "rpm2": rpm2,
            "wheel_radius_cm": WHEEL_RADIUS_CM,
            "wheel_dist_cm":   WHEEL_DIST_CM,
            "path": [{"x": n[0], "y": n[1], "theta_deg": n[2]} for n in path],
            "actions": [{"rpm_left": a[0], "rpm_right": a[1]} for a in actions_used]
        }
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "path_output.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Path exported → {json_path}")

    visualize(obs_raw, obs_inflated, start, goal, all_curves, path, rpm1, rpm2)


if __name__ == "__main__":
    main()
