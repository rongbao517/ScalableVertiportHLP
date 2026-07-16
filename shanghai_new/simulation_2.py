import argparse
import os
import pandas as pd
from generate_solution import regenerate_solution
from gurobi_optimization import run_gurobi_optimization
from initialization import initialize_states_with_time
from distance_battery import calculate_distance, battery_consumption_required, set_distance_data
from metrics import calculate_coverage_rate, calculate_cost, update_demand_chart
from task_assignment import time_step_path_assignment,update_arrivals    
from battery_charging import charging_and_battery_update, restore_vehicle_states,export_charging_log
from gurobi_optimization_Colum import column_generation
import time
# Global variable for unmet demand (list of tuples: (start, end, flow))
unmet_demand = []

def compute_most_needed(unmet_demand_local):
    if not unmet_demand_local:
        return ("", 0)
    need_dict = {}
    for s, e, flow in unmet_demand_local:
        need_dict[e] = need_dict.get(e, 0) + flow
    target = max(need_dict, key=need_dict.get)
    shortage = need_dict[target]
    return (target, shortage)

# def redistribute_vehicles(target, shortage, vertiport_states, plane_status, vehicle_states, vehicle_movements,
                        #   discharge_rate, cost_per_distance):
    from math import inf
    repositioned = 0
    total_added_cost = 0

    candidates = []
    for vehicle_id, status in plane_status.items():
        if status["status"] == "standby" and status["location"] != target:
            source = status["location"]
            if vehicle_states[vehicle_id]["loc"] == source:
                d = calculate_distance(source, target)
                candidates.append((vehicle_id, source, d))
    candidates.sort(key=lambda x: x[2])

    for vehicle_id, source, distance in candidates:
        if shortage <= 0:
            break
        required_battery = battery_consumption_required(distance, discharge_rate)
        if plane_status[vehicle_id]["battery"] >= required_battery:
            prev_location = plane_status[vehicle_id]["location"]
            plane_status[vehicle_id]["status"] = "in_service"
            plane_status[vehicle_id]["location"] = target
            plane_status[vehicle_id]["battery"] -= required_battery
            vehicle_states[vehicle_id]["loc"] = target
            vehicle_states[vehicle_id]["battery"] -= required_battery
            vehicle_states[vehicle_id]["in_service"] = 1
            vehicle_movements[vehicle_id] = (prev_location, target)

            # Update vertiport counts: departure from source and arrival at target
            vertiport_states[source]["avail"] = max(0, vertiport_states[source]["avail"] - 1)
            vertiport_states[source]["in_service"] += 1
            if vertiport_states[target]["in_service"] > 0:
                vertiport_states[target]["in_service"] -= 1
            vertiport_states[target]["avail"] += 1

            repositioned += 1
            shortage -= 1
            reposition_cost = distance * cost_per_distance
            total_added_cost += reposition_cost
    return repositioned, total_added_cost

def load_distance_map(distance_file):
    distance_matrix = pd.read_csv(distance_file, index_col=0)
    vertiports = distance_matrix.columns.tolist()
    distance_map = {}
    for i, start in enumerate(vertiports):
        for j, end in enumerate(vertiports):
            if i != j:
                distance_map[(start, end)] = distance_matrix.loc[start, end]
    return distance_map

def load_gurobi_results(file_path: str, time_step: int):
    data = pd.read_csv(file_path)
    grouped = data.groupby('Time')
    for time, group in grouped:
        if time == f"T{time_step}":
            time_step_results = []
            for _, row in group.iterrows():
                time_step_results.append({
                    "start": row["start"],
                    "end": row["end"],
                    "flow": int(row["flow"]),
                    "distance": float(row["distance"])
                })
            return time_step_results
    return []

def initialize_plane_status_loc(vehicles, vertiports, vehicles_number_each):
    plane_status = {}
    num_vertiports = len(vertiports)
    for i, vertiport in enumerate(vertiports):
        assigned_vehicles = vehicles[i * vehicles_number_each: (i + 1) * vehicles_number_each]
        for vehicle_id in assigned_vehicles:
            plane_status[vehicle_id] = {
                "battery": 100,
                "location": vertiport,
                "origin": vertiport,    # Save the initial location as origin
                "status": "standby",
                "idle_count": 0
            }
    return plane_status

def reset_plane_status(plane_status, vehicle_states, vertiport_states):
    """
    For vehicles that were in service last iteration, return them to their origin.
    Once they complete service (or repositioning), mark them as standby.
    """
    for vehicle_id, status in plane_status.items():
        if status["status"] == "in_service":
            origin = status.get("origin", status["location"])
            status["location"] = origin
            vehicle_states[vehicle_id]["loc"] = origin
            # Once in_service is complete, mark as standby.
            status["status"] = "standby"
            vehicle_states[vehicle_id]["in_service"] = 0
            if origin in vertiport_states:
                if vertiport_states[origin]["in_service"] > 0:
                    vertiport_states[origin]["in_service"] -= 1
                vertiport_states[origin]["avail"] += 1

# def calculate_demand_met(gurobi_results, vehicle_movements, current_unmet):
#     met_demand = sum(path["flow"] for path in gurobi_results)
#     # current_unmet reflects only this iteration's carried unmet demand (before assignment)
#     unmet_demand_amount = sum(flow for (_, _, flow) in current_unmet)
#     return met_demand, unmet_demand_amount

# def mandatory_return_assignment(plane_status, vehicle_states, vertiport_states, discharge_rate):
    """
    For each vehicle that is idle (standby) and not at its origin,
    if it has been away for one round, force a return.
    """
    # for vehicle_id, status in plane_status.items():
    #     if status["status"] == "standby" and status["location"] != status["origin"]:
    #         status["idle_count"] += 1
    #         if status["idle_count"] >= 1:  # Force return after one round away
    #             origin = status["origin"]
    #             current_loc = status["location"]
    #             dist = calculate_distance(current_loc, origin)
    #             battery_needed = battery_consumption_required(dist, discharge_rate)
    #             if status["battery"] >= battery_needed:
    #                 status["status"] = "in_service"
    #                 status["location"] = origin
    #                 status["battery"] -= battery_needed
    #                 vehicle_states[vehicle_id]["loc"] = origin
    #                 vehicle_states[vehicle_id]["battery"] -= battery_needed
    #                 vehicle_states[vehicle_id]["in_service"] = 1
    #                 if current_loc in vertiport_states:
    #                     vertiport_states[current_loc]["avail"] = max(0, vertiport_states[current_loc]["avail"] - 1)
    #                     vertiport_states[current_loc]["in_service"] += 1
    #                 if origin in vertiport_states:
    #                     if vertiport_states[origin]["in_service"] > 0:
    #                         vertiport_states[origin]["in_service"] -= 1
    #                     vertiport_states[origin]["avail"] += 1
    #                 status["idle_count"] = 0

def save_vehicle_states(vehicle_states, plane_status, iteration,k_value):   
    """
    Save the full state of each vehicle (location, battery, status, etc.)
    into a CSV file inside the folder 'vehicle_states'.
    """
    # folder = "vehicle_states"
    folder = f"vehicle_states_k{k_value}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    data = []
    for vehicle_id in vehicle_states:
        state = vehicle_states[vehicle_id]
        p_status = plane_status.get(vehicle_id, {})
        record = {
            "vehicle_id": vehicle_id,
            "location": state.get("loc", ""),
            "battery": state.get("battery", ""),
            "in_service": state.get("in_service", 0),
            "charging": state.get("charging", 0),
            "avail": state.get("avail", 0),
            "status": p_status.get("status", ""),
            "origin": p_status.get("origin", ""),
            "idle_count": p_status.get("idle_count", 0)
        }
        data.append(record)
    df = pd.DataFrame(data)
    df.to_csv(f"{folder}/iteration_{iteration}_k{k_value}.csv", index=False)
def calculate_operating_cost(vehicle_states, op_cost_per_vehicle):
    # 假设每辆车的运营成本为 op_cost_per_vehicle，每个车辆状态代表一天的成本
    return len(vehicle_states) * op_cost_per_vehicle
charging_log_tracker = {}
vertiport4_dispatch_log = []


from typing import List, Dict, Tuple
def build_total_paths(
    gurobi_flows: List[Dict],
    unmet_demand: List[Tuple[str, str, float]],
    distance_map: Dict[Tuple[str, str], float]
) -> List[Dict]:
    """
    合并当前 Gurobi 分配结果和上一轮 unmet_demand 为统一的调度输入格式。
    每个元素格式为：{"takeoff", "landing", "order_start", "order_end", "flow", "distance"}
    """
    unmet_as_paths = [
        {
            "takeoff": s,
            "landing": e,
            "order_start": s,
            "order_end": e,
            "flow": f,
            "distance": distance_map.get((s, e), 0)
        }
        for (s, e, f) in unmet_demand
    ]
    return gurobi_flows + unmet_as_paths



def run_iterations(num_iterations, vehicle_states, vertiport_states, gurobi_results_per_time, charging_rate,
                   discharge_rate, regenerate_solution, plane_status, distance_map, vertiports,vehicles_per_vertiport):
    global unmet_demand

    all_iteration_records = []
    time_step_summary_records = []
    cumulative_cost = 0  # cumulative overall cost
    debug_records = []
    all_node_metrics = []
    flight_depart_time = {}        # {plane_id: 上一次起飞的 k}
    vehicle_movements  = {}
    all_assigned_routes = []
    sankey_flows = []
    time_records = []   # ← 在循环外初始化列表


       




    for t in range(num_iterations):
        # Start with unmet_demand from previous iteration (carryover) and then clear the "current" values after logging
        current_unmet = unmet_demand.copy()  # this holds the unmet demand carried from previous round
        unmet_demand = []  # clear for the new iteration

        print(f"Time Step {t + 1}")


        def snapshot(tag):
            N_vehicles = len(vehicle_states)
            vertiport_capacity = {
                v: vertiport_states[v]["avail"] + vertiport_states[v]["in_service"]
                for v in vertiports if v in vertiport_states
            }
            total_capacity = sum(vertiport_capacity.values())
            count_in_service = sum(vs["in_service"] for vs in vehicle_states.values())
            count_standby = N_vehicles - count_in_service
            len_movs = len(vehicle_movements)
            print(
                f"[SNAP {tag}] "
                f"N_vehicles={N_vehicles}, "
                f"total_capacity={total_capacity}, "
                f"in_service={count_in_service}, "
                f"standby={count_standby}, "
                f"len_movs={len_movs}"
            )
            assert total_capacity == N_vehicles, f"车辆数不平衡({tag}): {total_capacity}!={N_vehicles}"
            assert len_movs == count_in_service, f"飞行中记录不一致({tag}): movs={len_movs}!={count_in_service}"

        # ———— 落地前快照 ————
        snapshot("before_update")
        update_arrivals(vehicle_states, vertiport_states, vehicle_movements,
                current_step=t, debug=True)
          # ———— 落地后快照 ————
        snapshot("after_update")

        restore_vehicle_states(vehicle_states, vehicle_movements)
        # reset_plane_status(plane_status, vehicle_states, vertiport_states)
        # vehicle_movements.clear()
        # First, assign vehicles based on previous unmet demand
                # 第一阶段：根据上一轮未满足需求分配车辆
        # unmet_after_assignment = time_step_path_assignment(
        #     gurobi_results_per_time[t], vehicle_states, vertiport_states, current_unmet, discharge_rate,
        #     vehicle_movements, plane_status
        # )
        # unmet_after_assignment, first_stage_assignments,launched_ids = time_step_path_assignment(
        #     gurobi_results_per_time[t], vehicle_states, vertiport_states, current_unmet,
        #     discharge_rate, vehicle_movements, plane_status
        # )

          # 3) 合并本轮“新订单” + “上一轮未满足”
        new_orders = [
        (row["start"], row["end"], float(row["flow"]))
        for row in gurobi_results_per_time[t]
        ]
        total_orders = current_unmet + new_orders

        # Debug：统计本轮的原始订单和未满足订单数量（个数 + 流量）
        raw_order_count = len(new_orders)
        raw_order_flow  = sum(flow for (_, _, flow) in new_orders)
        unmet_order_count = len(current_unmet)
        unmet_order_flow  = sum(flow for (_, _, flow) in current_unmet)

        print(f"[STAT] k={t} 原始订单数={raw_order_count}，流量={raw_order_flow}")
        print(f"[STAT] k={t} 上轮未满足订单数={unmet_order_count}，流量={unmet_order_flow}")
        print(f"[STAT] k={t} 总订单数={len(total_orders)}，总流量={sum(flow for (_, _, flow) in total_orders)}")
        # --- 3. 添加用于桑基图的记录 ---
        if unmet_order_flow > 0:
            sankey_flows.append({
                "source": "Unmet Last Round",
                "target": "Total Orders",
                "value": unmet_order_flow,
                "time_step": t
            })
        if raw_order_flow > 0:
            sankey_flows.append({
                "source": "New Orders",
                "target": "Total Orders",
                "value": raw_order_flow,
                "time_step": t
            })

    #     #   # 4) 一次性跑 Gurobi，把整个 total_orders 都丢给它
    #     gurobi_flows = run_gurobi_optimization(
    #         # t,
    #         # current_unmet,                # 旧的 unmet
    #         # gurobi_results_per_time[t],   # 新的订单
    #         # vertiports,
    #         # vehicles_per_vertiport,
    #         # distance_air
    #         t,
    #         current_unmet,
    #         gurobi_results_per_time[t],
    #         vertiports,
    #         vehicle_states,
    #         vehicles_per_vertiport,
    #         distance_air,
    #         discharge_rate,
    #         vertiport4_dispatch_log=vertiport4_dispatch_log, 
    #     )

    #     print(f"[DBG] k={t} Gurobi 流量: {gurobi_flows}")
       

    #     snapshot("before_assign")

    #     # 5) 构建派车输入路径（合并 Gurobi 和未满足）
    #     gurobi_flows = gurobi_results_per_time[t]
    #     gurobi_as_paths = [
    #     {
    #         "takeoff": row["takeoff"],        
    #         "landing": row["landing"],       
    #         "order_start": row["order_start"],
    #         "order_end": row["order_end"],
    #         "flow": float(row["flow"]),
    #         "distance": row["distance"]
    #     }
    #     for row in gurobi_flows if float(row["flow"]) > 0
    # ]
    #     total_paths = build_total_paths(
    #         gurobi_flows=gurobi_as_paths,
    #         unmet_demand=current_unmet,
    #         distance_map=distance_map
    #     )
    #     print(f"[合并路径] k={t} Gurobi={len(gurobi_flows)} 条, unmet={len(unmet_demand)} 条 → 合计 total_paths={len(total_paths)} 条")

        # original_orders = gurobi_results_per_time[t]

        # gurobi_results = run_gurobi_optimization(
        #     t,
        #     total_orders,
        #     vertiports,
        #     vehicle_states,
        #     vehicles_per_vertiport,
        #     distance_air,
        #     discharge_rate,
        #     vertiport4_dispatch_log=vertiport4_dispatch_log
        # )
        from collections import Counter
        vehicle_count = Counter(vs["loc"] for vs in vehicle_states.values())
        vehicle_total = sum(vehicle_count.values())
        # ——— 在调度调用前后打点 ———
        start = time.time()

        if not total_orders:
            print(f"[SKIP] T{t} 没有订单可优化，跳过 Gurobi 调用")
            gurobi_results = []
        else:
            # gurobi_results = run_gurobi_optimization(
            #     time_step=t,
            #     total_orders=total_orders,
            #     vertiports=vertiports,
            #     vehicle_states=vehicle_states,
            #     vehicles_per_vertiport=vehicles_per_vertiport,
            #     distance_air=distance_air,
            #     discharge_rate=discharge_rate,
            #     vehicle_count=vehicle_count,             
            #     vehicle_total=vehicle_total,
            #     vertiport4_dispatch_log=vertiport4_dispatch_log
            # )
            gurobi_results = column_generation(
                orders=total_orders,
                vertiports=vertiports,
                vehicle_states=vehicle_states,
                distance_air=distance_air,
                discharge_rate=discharge_rate,
                vehicle_count=vehicle_count,
                vehicle_total=vehicle_total,
                run_gurobi_optimization=run_gurobi_optimization,
                time_step=t,
                max_iters=1,
                max_pq_per_order=3
            )
        end = time.time()
        elapsed = end - start
        # 记录本次求解耗时
        time_records.append({
            "time_step": t,
            "solve_time": end - start
        })
        print(f"[Timestep {t}] 当前求解用时: {elapsed:.4f} 秒")
        # 记录订单流 → Gurobi分配
        sankey_flows.append({
            "source": "Total Orders",
            "target": "Gurobi Assigned",
            "value": sum(float(r["flow"]) for r in gurobi_results)
        })


        print(f"[DBG] k={t} Gurobi 分配路径数: {len(gurobi_results)}")

        gurobi_as_paths = [
            {
                "takeoff": row["takeoff"],
                "landing": row["landing"],
                "order_start": row["order_start"],
                "order_end": row["order_end"],
                "flow": float(row["flow"]),
                "distance": row["distance"]
            }
            for row in gurobi_results if float(row["flow"]) > 0
        ]

        total_paths = build_total_paths(
            gurobi_flows=gurobi_as_paths,
            unmet_demand=[],
            distance_map=distance_map
        )

        print(f"[合并路径] k={t} Gurobi={len(gurobi_as_paths)} 条, unmet={len(current_unmet)} 条 → 合计 total_paths={len(total_paths)} 条")

        snapshot("before_assign")

          # 5) 真·派车：把 Gurobi 给的 flows 一次性派出去
        unmet_after_assignment, assigned_routes, launched_ids = time_step_path_assignment(
            gurobi_results=total_paths,
            vehicle_states=vehicle_states,
            vertiport_states=vertiport_states,
            discharge_rate=discharge_rate,
            vehicle_movements=vehicle_movements,
            current_step=t,
            debug=True
        )
        actual_assigned = sum(r["flow"] for r in assigned_routes)
        unmet_total = sum(f for _, _, f in unmet_after_assignment)

        sankey_flows.append({
            "source": "Gurobi Assigned",
            "target": "Actually Dispatched",
            "value": actual_assigned
        })
        sankey_flows.append({
            "source": "Gurobi Assigned",
            "target": "Unmet After Dispatch",
            "value": unmet_total
        })

        for route in assigned_routes:
            all_assigned_routes.append({
            "time_step": t + 1,
            "takeoff": route["start"],
            "landing": route["end"],
            "flow": route["flow"],
            "distance": route["distance"]
        })

        print(f"[DBG] k={t} 派车完成，launched={len(launched_ids)} 架")
        snapshot("after_assign")
        from collections import Counter
        plane_locations = Counter(vs["loc"] for vs in vehicle_states.values())
        print(f"[T={t}] 车辆分布: {dict(plane_locations)}")

        # 6) 更新起飞时间
        for vid in launched_ids:
            flight_depart_time[vid] = t

        # 7) carry‐over，下一轮的 unmet
        unmet_demand = unmet_after_assignment
        print(f"[DBG] k={t} 下一轮 carry_unmet: {unmet_demand}")

        # --- 下面是数据统计、计费等，改成单阶段即可 ---
        # total_demand = sum(flow for (_, _, flow) in total_orders)
        # remaining    = sum(flow for (_, _, flow) in unmet_after_assignment)
        # met_qty      = total_demand - remaining
        met_qty = sum(route["flow"] for route in assigned_routes)
        # remaining = sum(flow for (_, _, flow) in unmet_after_assignment)

        total_demand = sum(flow for (_, _, flow) in total_orders)
        remaining = total_demand-met_qty



        # 只用一次 cost 计算
        transport_cost = calculate_cost(
            assigned_routes, cost_per_distance=4, distance_map=distance_map
        )
        operating_cost = calculate_operating_cost(
            vehicle_states, op_cost_per_vehicle=5
        )
        iteration_cost = transport_cost + operating_cost
        cumulative_cost += iteration_cost

        # 记录单阶段信息
        one_stage_info = {
            "total_demand": total_demand,
            "remaining": remaining,
            "met": met_qty,
            "transport_cost": transport_cost,
            "operating_cost": operating_cost,
            "iteration_cost": iteration_cost
        }


        # unmet_after_assignment, first_stage_assignments, launched_ids = time_step_path_assignment(
        #     gurobi_results_per_time[t],
        #     vehicle_states,
        #     vertiport_states,
        #     current_unmet,
        #     discharge_rate,
        #     vehicle_movements,
        #     current_step = t,        # <— new
        #     debug        = True      # <— optional
        #     )
        # print(f"[DBG] k={t}  可派 out={len(launched_ids)} ids={launched_ids[:5]}")

        # if gurobi_results_per_time[t]:
        #     start = gurobi_results_per_time[t][0]["start"]
        #     # 距离和电耗门槛
        #     bat_req = battery_consumption_required(
        #         gurobi_results_per_time[t][0]["distance"], discharge_rate
        #     )
        #     # 本地所有待命车辆
        #     local_standby = [
        #         vid for vid, vs in vehicle_states.items()
        #         if vs["loc"] == start and vs["in_service"] == 0
        #     ]
        #     # 本地待命但电量不足
        #     lowbat = [
        #         vid for vid in local_standby
        #         if vehicle_states[vid]["battery"] < bat_req
        #     ]
        #     # 本地电量够但没被 dispatch（不在 launched_ids） 的车辆
        #     other_filt = [
        #         vid for vid in local_standby
        #         if vid not in launched_ids and vid not in lowbat
        #     ]
        #     # 本地正在服务（busy）
        #     busy = sum(
        #         1 for vs in vehicle_states.values()
        #         if vs["loc"] == start and vs["in_service"] == 1
        #     )

        #     print(
        #         f"[DBG] k={t} {start} 本地待命={len(local_standby)} busy={busy}  "
        #         f"可派={len(launched_ids)}  lowbat={len(lowbat)}  filtered={len(other_filt)}  "
        #         f"IDs_filtered={other_filt[:5]}..."
        #     )
        # # 车辆起飞时间更新 

        # for vid in launched_ids:
        #     flight_depart_time[vid] = t
        # print(f"[Debug] Time Step {t+1} - time_step_path_assignment 返回 unmet_after_assignment: {unmet_after_assignment}")

        # # 计算第一阶段总需求：当前时刻新需求 + 上一轮遗留 unmet demand
        # total_demand_first_stage = sum(path["flow"] for path in gurobi_results_per_time[t]) + \
        #                            sum(flow for (_, _, flow) in current_unmet)
        # print(f"[Debug] Time Step {t+1} - total_demand_first_stage: {total_demand_first_stage}")

        # # 计算第一阶段剩余需求
        # remaining_first_stage = sum(flow for (_, _, flow) in unmet_after_assignment)
        # print(f"[Debug] Time Step {t+1} - remaining_first_stage: {remaining_first_stage}")

        # # 第一阶段满足的需求量
        # met_demand_assignment = total_demand_first_stage - remaining_first_stage
        # print(f"[Debug] Time Step {t+1} - met_demand_assignment: {met_demand_assignment}")

        # first_stage_cost = calculate_cost(first_stage_assignments, cost_per_distance=4, distance_map=distance_map)
        # print(f"[Debug] Time Step {t+1} - first_stage_cost: {first_stage_cost}")
        #  # 保存第一阶段调试信息
        # first_stage_info = {
        #     "total_demand_first_stage": total_demand_first_stage,
        #     "remaining_first_stage": remaining_first_stage,
        #     "met_demand_assignment": met_demand_assignment,
        #     "first_stage_cost": first_stage_cost,
        # }

        # # 第二阶段：针对第一阶段未满足的需求进行 Gurobi 优化
        # gurobi_results, unmet_from_optimization = run_gurobi_optimization(
        #     t, unmet_after_assignment, [], vertiports, vehicles_per_vertiport, distance_air
        # )
        # print(f"[Debug] Time Step {t+1} - run_gurobi_optimization 返回 gurobi_results: {gurobi_results}")
        # print(f"[Debug] Time Step {t+1} - run_gurobi_optimization 返回 unmet_from_optimization: {unmet_from_optimization}")

        # total_demand_second_stage = sum(flow for (_, _, flow) in unmet_after_assignment)
        # print(f"[Debug] Time Step {t+1} - total_demand_second_stage: {total_demand_second_stage}")

        # remaining_second_stage = sum(flow for (_, _, flow) in unmet_from_optimization)
        # print(f"[Debug] Time Step {t+1} - remaining_second_stage: {remaining_second_stage}")

        # met_demand_optimization = total_demand_second_stage - remaining_second_stage
        # print(f"[Debug] Time Step {t+1} - met_demand_optimization: {met_demand_optimization}")

        #  # 计算第二阶段成本
        # second_stage_cost = calculate_cost(gurobi_results, cost_per_distance=4, distance_map=distance_map)
        # print(f"[Debug] Time Step {t+1} - second_stage_cost: {second_stage_cost}")
        # second_stage_info = {
        #     "total_demand_second_stage": total_demand_second_stage,
        #     "remaining_second_stage": remaining_second_stage,
        #     "met_demand_optimization": met_demand_optimization,
        #     "second_stage_cost": second_stage_cost,
        # }

        # # iteration_cost = first_stage_cost + second_stage_cost
        # operating_cost = calculate_operating_cost(vehicle_states, op_cost_per_vehicle=5)  # 举例 0.1单位成本
        # iteration_cost = first_stage_cost + second_stage_cost + operating_cost
        # print(f"[Debug] Time Step {t+1} - iteration_cost: {iteration_cost}")
        # # 总满足需求为两阶段的和；未满足需求则为第二阶段剩余
        # cumulative_cost += iteration_cost
        # met_demand_iter = met_demand_assignment + met_demand_optimization
        # unmet_demand_iter = unmet_from_optimization
        # print(f"[Debug] Time Step {t+1} - met_demand_iter: {met_demand_iter}, unmet_demand_iter: {unmet_demand_iter}")

        # # 更新全局未满足需求，供下一轮使用
        # unmet_demand = unmet_from_optimization
        # print(f"[Debug] Time Step {t+1} - 更新全局 unmet_demand: {unmet_demand}")
         # 重新定位车辆步骤（可选，根据你的需求来判断是否需要记录）
        target, shortage_val = compute_most_needed(unmet_demand)
        # redistribution_info = {"repositioned": 0, "redistribution_added_cost": 0}
        # if target and shortage_val > 0:
        #     repositioned, added_cost = redistribute_vehicles(
        #         target, shortage_val, vertiport_states, plane_status,
        #         vehicle_states, vehicle_movements,
        #         discharge_rate, cost_per_distance=4
        #     )
            # iteration_cost += added_cost
            # cumulative_cost += added_cost
            # # redistribution_info = {"repositioned": repositioned, "redistribution_added_cost": added_cost}
            # print(f"[Debug] Time Step {t+1} - redistribution: repositioned={repositioned}, added_cost={added_cost}")
        # # Calculate met/unmet for this assignment step
        # met_demand_assignment, unmet_demand_assignment = calculate_demand_met(gurobi_results_per_time[t], vehicle_movements, current_unmet)

        # # Next, run Gurobi optimization to handle new demand and any remaining unmet demand
        # gurobi_results, unmet_from_optimization = run_gurobi_optimization(t, unmet_after_assignment, gurobi_results_per_time[t], vertiports)
        # # The Gurobi results here are additional assignments for this iteration.
        # # Calculate met/unmet for Gurobi part
        # met_demand_optimization, unmet_demand_optimization = calculate_demand_met(gurobi_results, vehicle_movements, unmet_after_assignment)

        # # The met demand for this iteration is the sum from both assignment steps.
        # met_demand_iter = met_demand_assignment + met_demand_optimization
        # # The unmet demand for this iteration is ONLY what remains from the optimization
        # unmet_demand_iter = unmet_demand_optimization

        # # Ensure unmet_demand_iter remains for the next iteration (carried over)
        # unmet_demand = unmet_from_optimization

        # Compute iteration cost and update cumulative cost
        # iteration_cost = calculate_cost(
        #     flow_data=gurobi_results,
        #     cost_per_distance=4,
        #     distance_map=distance_map
        # )
        

        # Recalculate vertiport vehicle counts based on current vehicle states
         # 更新车辆状态后，重新统计停机坪状态
       

        vertiport_tracking = {
            v: (vertiport_states[v].get("avail", 0) + vertiport_states[v].get("in_service", 0))
            for v in vertiports if v in vertiport_states
        }

        # 计算当前迭代的节点级指标
        node_metrics = {}
        for v in vertiports:
            avail = vertiport_states[v].get("avail", 0)
            in_service = vertiport_states[v].get("in_service", 0)
            total_vehicles = avail + in_service
            # 使用 current_unmet 计算该节点未满足订单（这里 current_unmet 是上一轮的未满足需求）
            node_demand = sum(flow for start, end, flow in current_unmet if start == v)
            balance = total_vehicles - node_demand
            node_metrics[v] = {
                "avail": avail,
                "in_service": in_service,
                "total_vehicles": total_vehicles,
                "node_demand": node_demand,
                "balance": balance
            }

        # # 构建当前迭代的调试信息，包含全局指标与节点级指标
        # iteration_debug = {}
        # iteration_debug["time_step"] = t + 1
        # iteration_debug["first_stage"] = first_stage_info
        # iteration_debug["second_stage"] = second_stage_info
        # iteration_debug["iteration_cost"] = iteration_cost
        # iteration_debug["cumulative_cost"] = cumulative_cost
        # iteration_debug["vertiport_tracking"] = vertiport_tracking
        # iteration_debug["node_metrics"] = node_metrics

        # debug_records.append(iteration_debug)

        # # 记录全局 summary 记录，保留你需要的字段
        # # 同时计算 total_unmet 全局未满足订单总量（这里用 unmet_demand_iter）
        # total_unmet = sum(flow for (_, _, flow) in unmet_demand_iter) if unmet_demand_iter else 0

        # record = {
        #     "time_step": t + 1,
        #     "met_demand": met_demand_iter,
        #     "unmet_demand": total_unmet,
        #     "iteration_cost": iteration_cost,
        #     "cumulative_cost": cumulative_cost,
        #     "vertiport_counts": vertiport_tracking
        # }
        # all_iteration_records.append(record)
        # time_step_summary_records.append(record)

         # 构建本轮调试信息
        iteration_debug = {
            "time_step":      t + 1,
            "stage_info":     one_stage_info,
            "iteration_cost": iteration_cost,
            "cumulative_cost": cumulative_cost,
            "vertiport_tracking": vertiport_tracking,
            "node_metrics":   node_metrics
        }
        debug_records.append(iteration_debug)

        # 汇总记录
        record = {
            "time_step":     t + 1,
            "met_demand":    met_qty,
            "unmet_demand":  remaining,
            "iteration_cost": iteration_cost,
            "cumulative_cost": cumulative_cost,
            "vertiport_counts": vertiport_tracking
        }
        all_iteration_records.append(record)
        time_step_summary_records.append(record)

        # 将当前迭代的节点指标写入全局节点记录列表
        for v in vertiports:
            node_record = {
                "time_step": t + 1,
                "vertiport": v,
                "avail": node_metrics[v]["avail"],
                "in_service": node_metrics[v]["in_service"],
                "total_vehicles": node_metrics[v]["total_vehicles"],
                "node_demand": node_metrics[v]["node_demand"],
                "balance": node_metrics[v]["balance"]
            }
            all_node_metrics.append(node_record)


        # Optionally, reposition vehicles if there is a shortage (not affecting met/unmet metrics)
        target, shortage = compute_most_needed(unmet_demand)
        # if target and shortage > 0:
        #     repositioned, added_cost = redistribute_vehicles(target, shortage, vertiport_states, plane_status,
        #                                                       vehicle_states, vehicle_movements,
        #                                                       discharge_rate, cost_per_distance=4)
        #     iteration_cost += added_cost
        #     cumulative_cost += added_cost

        # Update vehicle charging; fully charged vehicles become standby for the next iteration
        # charging_and_battery_update(vehicle_states, time_interval=15, charging_rate=charging_rate)
        charging_and_battery_update(
            vehicle_states,
            time_interval=15,
            charging_rate=charging_rate,
            current_step=t,
            charging_tracker=charging_log_tracker
)
        for vid, state in vehicle_states.items():
            if state["battery"] >= 100:
                print(f"[检查点] {vid} 已充满，但 charging 状态为 {state['charging']}，avail 状态为 {state['avail']}")

        # Save current vehicle state snapshot
        save_vehicle_states(vehicle_states, plane_status, t + 1,k_value=new_k)
    
    import pandas as pd
    import pandas as pd
    # # pd.DataFrame(vertiport4_dispatch_log).to_csv("vertiport4_dispatch.csv", index=False)
    pd.DataFrame(all_assigned_routes).to_csv("1_column_path_dispatch_records_10.csv", index=False)
    # pd.DataFrame(all_assigned_routes).to_csv("1__path_dispatch_records_10.csv", index=False)
    print("路径分配数据已保存为 path_dispatch_records.csv")
    # pd.DataFrame(sankey_flows).to_csv("sankey_order_flow_summary.csv", index=False)
    # print("Sankey Diagram 数据已保存为 sankey_order_flow_summary.csv")
    df_time = pd.DataFrame(time_records)
    print(f"平均求解时间: {df_time['solve_time'].mean():.4f} s")
    print(f"累计求解时间: {df_time['solve_time'].sum():.4f} s")



    
    

    # # 将嵌套字典转换为平坦结构
    # df = pd.json_normalize(debug_records)

    # # 保存为 CSV 文件
    # # df.to_csv("test_debug_records_10_1.csv", index=False)
    suffix = f"1_column_k{new_k}_charging{charging_rate}_discharge{discharge_rate}_vehicles{vehicles_number_each}"
    # suffix = f"1__k{new_k}_charging{charging_rate}_discharge{discharge_rate}_vehicles{vehicles_number_each}"
    df = pd.json_normalize(debug_records)
    # df.to_csv(f"1__test_debug_records_{suffix}.csv", index=False)
    df.to_csv(f"1_column_test_debug_records_{suffix}.csv", index=False)
    
    print(f"调试信息已保存为 test_debug_records_{suffix}.csv")

    print("调试信息已保存为 CSV 格式：debug_records.csv")
    # df_nodes = pd.DataFrame(all_node_metrics)
    # df_nodes.to_csv(f"test_node_metrics_k{new_k}.csv", index=False)
    df_nodes = pd.DataFrame(all_node_metrics)
    # df_nodes.to_csv(f"1__test_node_metrics_{suffix}.csv", index=False)
    df_nodes.to_csv(f"1_column_test_node_metrics_{suffix}.csv", index=False)
    print(f"节点级指标已保存为 test_node_metrics_{suffix}.csv")

    print(f"节点级指标已保存为 node_metrics_k{new_k}.csv")
    export_charging_log(charging_log_tracker)
    print("充电日志已保存为 charging_log.csv")


    # Save detailed iteration metrics to CSV file "detail.csv"
    # detail_df = pd.DataFrame(time_step_summary_records)
    # detail_df.to_csv("detail.csv", index=False)
    detail_df = pd.DataFrame(time_step_summary_records)
    # detail_df.to_csv(f"1__detail_{suffix}.csv", index=False)
    detail_df.to_csv(f"1_column_detail_{suffix}.csv", index=False)
    print(f"All iteration records saved to detail_{suffix}.csv")

    print("All iteration records saved to detail.csv")

    # assert all(
    #     vs["avail"] + vs["in_service"] == vehicles_per_vertiport
    #     for vs in vertiport_states.values()
    # ), f"车辆计数不平衡 @ k={t}"

    return all_iteration_records, time_step_summary_records, cumulative_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertiports_file", default="./Adjust/15/vertiports_15.csv")
    parser.add_argument("--distance_file", default="./Adjust/15/distance_15.csv")
    parser.add_argument("--gurobi_results_file", default="./Adjust/15/sequential_optimized_results_with_integer_flow_cluster_results_15.csv")
    parser.add_argument("--k_value", type=int, default=15, help="当前实验的 k 值")

    args = parser.parse_args()
    new_k = args.k_value  # 使用命令行参数传入的 k 值
    # 在主程序中调用 set_distance_data() 加载距离数据
    set_distance_data(args.distance_file)


    vertiports_df = pd.read_csv(args.vertiports_file)
    vertiports = vertiports_df["Vertiport"].tolist()
    distance_map = load_distance_map(args.distance_file)
    # 在主函数中加载距离数据
    distance_matrix = pd.read_csv(args.distance_file, index_col=0)
    def get_distance_air(distance_matrix):
        return {
        (p, q): float(distance_matrix.loc[p, q])
        for p in distance_matrix.index
        for q in distance_matrix.columns
        if p != q
    }

    distance_air = get_distance_air(distance_matrix)
    gurobi_results_per_time = []
    total_time_steps = 500  # Set number of iterations as needed
    for t in range(total_time_steps):
        gurobi_results = load_gurobi_results(args.gurobi_results_file, t)
        gurobi_results_per_time.append(gurobi_results)

    global_time_step_records = []
    results = []

    # charging_rates = [30, 35, 40]
    # discharge_rates = [1, 3, 5]
    # vehicle_counts = [10, 15, 20,100]


    charging_rates = [3.5]
    discharge_rates = [1]
    # vehicle_counts = [10,15,20,100]
    vehicle_counts = [5]
    new_k = 15
    for charging_rate in charging_rates:
        for discharge_rate in discharge_rates:
            for vehicles_number_each in vehicle_counts:
                # global vehicles_per_vertiport
                # vehicles_per_vertiport = vehicles_number_each
                vehicles = ["V" + str(i) for i in range(1, vehicles_number_each * len(vertiports) + 1)]
                vehicle_states, vertiport_states = initialize_states_with_time(vehicles, vertiports, vehicles_number_each)
                plane_status = initialize_plane_status_loc(vehicles, vertiports, vehicles_number_each)
                for vertiport in vertiports:
                    vertiport_states[vertiport]["activated"] = True

                unmet_demand = []
                all_iteration_records, time_step_summary_records, cumulative_cost = run_iterations(
                    num_iterations=500,
                    vehicle_states=vehicle_states,
                    vertiport_states=vertiport_states,
                    gurobi_results_per_time=gurobi_results_per_time,
                    charging_rate=charging_rate,
                    discharge_rate=discharge_rate,
                    regenerate_solution=regenerate_solution,
                    plane_status=plane_status,
                    distance_map=distance_map,
                    vertiports=vertiports,
                    vehicles_per_vertiport=vehicles_number_each
                )

                # coverage_rate = (sum(rec["met_demand"] for rec in time_step_summary_records) /
                #                  sum(rec["met_demand"] + rec["unmet_demand"] for rec in time_step_summary_records)) if time_step_summary_records else 0

                # coverage_rate = (sum(rec["met_demand"] for rec in time_step_summary_records) /sum(rec["met_demand"] + sum(flow for (_, _, flow) in rec["unmet_demand"]) for rec in time_step_summary_records)) if time_step_summary_records else 0
                coverage_rate = (sum(rec["met_demand"] for rec in time_step_summary_records) /(sum(rec["met_demand"] for rec in time_step_summary_records) + sum(rec["unmet_demand"] for rec in time_step_summary_records))) if time_step_summary_records else 0


                # 假设 new_k 表示当前使用的停机坪数量（例如 new_k = k）
                cost_per_vertiport = 73907  # 每个停机坪的建设成本（美元）
                lifetime_in_days = 4745      # 使用寿命（例如10年，共3650天）
                daily_infra_cost_per_vertiport = cost_per_vertiport / lifetime_in_days  # 每个停机坪每天的摊销成本
                total_infra_daily_cost = new_k * daily_infra_cost_per_vertiport  # 所有停机坪每天的基础设施成本

            # 假设 cumulative_cost 表示仿真得到的每天的累计运营成本（运营成本部分）
                total_daily_cost = cumulative_cost + total_infra_daily_cost
                total_cost = cumulative_cost

                results.append({
                    "charging_rate": charging_rate,
                    "discharge_rate": discharge_rate,
                    "vehicle_count": vehicles_number_each,
                    "coverage_rate": coverage_rate,
                    "cumulative_cost": total_cost
                })

                details_df = pd.DataFrame(all_iteration_records)
                
                details_filename = f"sequential_experiment_details_k{new_k}_charging{charging_rate}_discharge{discharge_rate}_vehicles{vehicles_number_each}.csv"

                # details_filename = f"experiment_details_charging{charging_rate}_discharge{discharge_rate}_vehicles{vehicles_number_each}.csv"
                details_filepath = os.path.join("detail_csv", details_filename)
                details_df.to_csv(details_filepath, index=False)

                for record in time_step_summary_records:
                    record["charging_rate"] = charging_rate
                    record["discharge_rate"] = discharge_rate
                    record["vehicle_count"] = vehicles_number_each
                    global_time_step_records.append(record)

    # df = pd.DataFrame(results)
    # df.to_csv("sensitivity_analysis_results.csv", index=False)
    sensitivity_filename = f"test_sensitivity_analysis_results_k{new_k}_vehicles{vehicles_number_each}.csv"

    df = pd.DataFrame(results)
    df.to_csv(sensitivity_filename, index=False)
    print("Overall experiment results saved to sensitivity_analysis_results.csv")

    # detail_df = pd.DataFrame(global_time_step_records)
    # detail_df.to_csv("detail.csv", index=False)
    # global_detail_filename = f"detail_k{new_k}.csv"
    # detail_df = pd.DataFrame(global_time_step_records)
    # detail_df.to_csv(global_detail_filename, index=False)
    # print("All iteration records saved to detail.csv")

    # global_detail_filename = f"filiter_re_k{new_k}.csv"
    # detail_df = pd.DataFrame(global_time_step_records)
    # detail_df.to_csv(global_detail_filename, index=False)
    # print("All iteration records saved to detail.csv")

    import os
    # 假设 charging_rate, discharge_rate, vehicles_number_each, new_k 为循环外的最后值（或者你只运行了一组参数）
    global_detail_filename = f"test_inter_k{new_k}_charging{charging_rate}_discharge{discharge_rate}_vehicles{vehicles_number_each}.csv"


    # 定义文件夹名称
    # folder = "results"
    folder = "test_seq_redistubution_results"

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 构造最终保存的文件路径
    filepath = os.path.join(folder, global_detail_filename)
    detail_df = pd.DataFrame(global_time_step_records)

    # 保存 DataFrame 到 CSV 文件
    detail_df.to_csv(filepath, index=False)
    print(f"All iteration records saved to {filepath}")


# run the draw.py file afterwards to visualize the results
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Read the new detail CSV file
# # df = pd.read_csv('detail.csv')

# # read files from folder
# import glob
# import os
# for file in glob.glob("detail_csv/*.csv"):
#     print(file)
#     df = pd.read_csv(file)
#     print(df['unmet_demand'].head())

#     # 将 "unmet_demand" 列转换为数值类型
#     df['unmet_demand'] = pd.to_numeric(df['unmet_demand'], errors='coerce')

#     # If assignment_ratio is missing, compute it as met_demand / (met_demand + unmet_demand)
#     if 'assignment_ratio' not in df.columns:
#         df['assignment_ratio'] = df.apply(lambda row: row['met_demand'] / (row['met_demand'] + row['unmet_demand'])
#                                         if (row['met_demand'] + row['unmet_demand']) != 0 else 0, axis=1)

#     # If big_picture_assignment_ratio is missing and real_flow exists, compute it similarly
#     if 'big_picture_assignment_ratio' not in df.columns and 'real_flow' in df.columns:
#         df['big_picture_assignment_ratio'] = df.apply(lambda row: row['met_demand'] / row['real_flow']
#                                                     if row['real_flow'] != 0 else 0, axis=1)
#     elif 'big_picture_assignment_ratio' not in df.columns:
#         # Otherwise, default to assignment_ratio if real_flow is not present
#         df['big_picture_assignment_ratio'] = df['assignment_ratio']

#     # Display a preview of the data
#     print("Data Preview:")
#     print(df.head())

    # Construct a summary paragraph describing key parameters
    # summary = (
    #     f"The dataset spans {df['time_step'].nunique()} time steps and includes several key performance indicators. "
    #     f"It tracks the cumulative cost incurred at each time step as well as the cost added per iteration (iteration_cost). "
    #     f"Operational performance is captured by the met_demand (fulfilled orders) and unmet_demand (orders left unfulfilled). "
    #     f"Assignment performance is recorded by the assignment_ratio and big_picture_assignment_ratio. "
    #     f"Additional parameters include a constant charging rate of {df['charging_rate'].iloc[0]} and a discharge rate of {df['discharge_rate'].iloc[0]}, "
    #     f"with the fleet comprising {df['vehicle_count'].iloc[0]} vehicles."
    # )
    # print("\nSummary of the Dataset:")
    # print(summary)

    # Create directory for figures if it doesn't exist
    # file_name = os.path.basename(file)
    # file_name = file_name.split('.')[0]
    # folder_path = 'figure' + file_name
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # # Set plot style
    # plt.style.use('seaborn-v0_8-paper')

    # # 1. Line Plot: Cumulative Cost, Iteration Cost, Met Demand, and Unmet Demand Over Time
    # plt.figure(figsize=(12, 8))
    # plt.plot(df['time_step'], df['cumulative_cost'], marker='o', label='Cumulative Cost')
    # plt.plot(df['time_step'], df['iteration_cost'], marker='s', label='Iteration Cost')
    # plt.plot(df['time_step'], df['met_demand'], marker='^', label='Met Demand')
    # plt.plot(df['time_step'], df['unmet_demand'], marker='d', label='Unmet Demand')
    # plt.title('Operational Metrics Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Cost / Demand')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(folder_path + '/line_plot_operational_metrics.png')
    # # plt.show()

    # # 2. Scatter Plot: Time vs Assignment Ratio
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df['time_step'], df['assignment_ratio'], c='blue', edgecolor='k', alpha=0.7)
    # plt.title('Assignment Ratio Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Assignment Ratio')
    # plt.tight_layout()
    # plt.savefig(folder_path  + '/scatter_assignment_ratio.png')
    # # plt.show()

    # 3. Bar Plot: Charging Rate and Discharge Rate
    # rates = {'Charging Rate': df['charging_rate'].iloc[0], 'Discharge Rate': df['discharge_rate'].iloc[0]}
    # plt.figure(figsize=(6, 6))
    # plt.bar(list(rates.keys()), list(rates.values()), color=['skyblue', 'salmon'], edgecolor='black')
    # plt.title('Charging vs Discharge Rate')
    # plt.ylabel('Rate')
    # plt.tight_layout()
    # plt.savefig('figure/bar_rates.png')
    # plt.show()

    # 4. Histogram: Distribution of Cumulative Cost and Iteration Cost
    # plt.figure(figsize=(12, 6))
    # plt.hist(df['cumulative_cost'], bins=10, edgecolor='black', alpha=0.7, label='Cumulative Cost')
    # plt.hist(df['iteration_cost'], bins=10, edgecolor='black', alpha=0.7, label='Iteration Cost')
    # plt.title('Distribution of Cumulative and Iteration Costs')
    # plt.xlabel('Cost')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(folder_path + '/hist_costs.png')
    # # plt.show()

    # # 5. Box Plot: Cumulative Cost by Time Step
    # plt.figure(figsize=(12, 6))
    # sns.boxplot(x=df['time_step'], y=df['cumulative_cost'])
    # plt.title('Box Plot of Cumulative Cost by Time Step')
    # plt.xlabel('Time Step')
    # plt.ylabel('Cumulative Cost')
    # plt.tight_layout()
    # plt.savefig(folder_path + '/boxplot_cumulative_cost.png')
    # # plt.show()

    # # 6. Line Plot: Demand Trends Over Time (Real Flow, Met Demand, Unmet Demand)
    # plt.figure(figsize=(12, 8))
    # if 'real_flow' in df.columns:
    #     plt.plot(df['time_step'], df['real_flow'], marker='o', label='Real Flow (Total Demand)')
    # plt.plot(df['time_step'], df['met_demand'], marker='s', label='Met Demand')
    # plt.plot(df['time_step'], df['unmet_demand'], marker='^', label='Unmet Demand')
    # plt.title('Demand Trends Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Demand')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(folder_path + '/line_plot_demand_trends.png')
    # # plt.show()

    # # 7. Line Plot: Assignment Ratios Over Time
    # plt.figure(figsize=(12, 8))
    # plt.plot(df['time_step'], df['assignment_ratio'], marker='o', label='Assignment Ratio')
    # plt.plot(df['time_step'], df['big_picture_assignment_ratio'], marker='s', label='Big Picture Assignment Ratio')
    # plt.title('Assignment Ratios Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Ratio')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(folder_path + '/line_plot_assignment_ratios.png')
    # # plt.show()

    # # 8. Correlation Heatmap: Key Numeric Metrics
    # numeric_cols = ['cumulative_cost', 'iteration_cost', 'met_demand', 'unmet_demand', 'assignment_ratio', 'big_picture_assignment_ratio']
    # if 'real_flow' in df.columns:
    #     numeric_cols.append('real_flow')
    # corr = df[numeric_cols].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Heatmap of Key Metrics')
    # plt.tight_layout()
    # plt.savefig(folder_path + '/heatmap_metrics.png')
    # # plt.show()

    # # 9. Pair Plot: Explore Relationships Between Key Metrics
    # try:
    #     sns.pairplot(df[numeric_cols])
    #     plt.suptitle('Pair Plot of Key Metrics', y=1.02)
    #     plt.tight_layout()
    #     plt.savefig(folder_path + '/pairplot_metrics.png')
    #     # plt.show()
    # except Exception as e:
    #     print("Pair plot could not be generated:", e)
