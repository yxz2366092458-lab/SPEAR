#!/usr/bin/env python3
"""
生成2x2交通灯网格网络的完整配置文件
"""

import os
import subprocess
import xml.etree.ElementTree as ET
import random


def generate_network():
    """生成2x2网格网络"""

    # 使用netgenerate生成基础网格
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.number", "2",  # 2x2网格
        "--grid.length", "200",  # 每条边长度200米
        "--output-file", "grid_2x2.net.xml",
        "--default-junction-type", "traffic_light",
        "--tls.guess", "true",  # 自动设置交通灯
        "--tls.join", "true",  # 合并交通灯逻辑
        "--grid.attach-length", "50",  # 边缘连接长度
        "--no-turnarounds", "true",  # 禁止掉头
        "--no-internal-links", "false"
    ]

    print("正在生成2x2网格网络...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"生成网络时出错: {result.stderr}")
        # 如果netgenerate不可用，创建简单网络XML
        return create_simple_network_xml()

    print("✅ 网络文件已生成: grid_2x2.net.xml")
    return "grid_2x2.net.xml"


def create_simple_network_xml():
    """手动创建2x2网格网络XML（备用方法）"""

    root = ET.Element("net")
    root.set("version", "1.9")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/net_file.xsd")

    # 位置信息
    location = ET.SubElement(root, "location")
    location.set("netOffset", "0.00,0.00")
    location.set("convBoundary", "0.00,0.00,500.00,500.00")
    location.set("origBoundary", "0.00,0.00,500.00,500.00")
    location.set("projParameter", "!")

    # 道路类型
    type_elem = ET.SubElement(root, "type")
    type_elem.set("id", "highway.urban")
    type_elem.set("numLanes", "1")
    type_elem.set("speed", "13.89")  # 50 km/h

    # 2x2网格的4个节点 (交叉口)
    nodes = [
        ("node0", 100, 100, "traffic_light"),
        ("node1", 300, 100, "traffic_light"),
        ("node2", 100, 300, "traffic_light"),
        ("node3", 300, 300, "traffic_light")
    ]

    for node_id, x, y, ntype in nodes:
        node = ET.SubElement(root, "junction")
        node.set("id", node_id)
        node.set("x", str(x))
        node.set("y", str(y))
        node.set("type", ntype)
        if ntype == "traffic_light":
            node.set("tl", node_id)

    # 边 (道路) - 水平方向
    horizontal_edges = [
        ("edge_h0", "node0", "node1", 200),
        ("edge_h1", "node1", "node0", 200),
        ("edge_h2", "node2", "node3", 200),
        ("edge_h3", "node3", "node2", 200)
    ]

    # 边 (道路) - 垂直方向
    vertical_edges = [
        ("edge_v0", "node0", "node2", 200),
        ("edge_v1", "node2", "node0", 200),
        ("edge_v2", "node1", "node3", 200),
        ("edge_v3", "node3", "node1", 200)
    ]

    # 外部连接边（让车辆可以进出）
    external_edges = [
        ("edge_in0", "bottom_in", "node0", 50),
        ("edge_out0", "node0", "bottom_out", 50),
        ("edge_in1", "right_in", "node1", 50),
        ("edge_out1", "node1", "right_out", 50),
        ("edge_in2", "left_in", "node2", 50),
        ("edge_out2", "node2", "left_out", 50),
        ("edge_in3", "top_in", "node3", 50),
        ("edge_out3", "node3", "top_out", 50)
    ]

    # 添加外部节点
    external_nodes = [
        ("bottom_in", 100, 0, "priority"),
        ("bottom_out", 100, 0, "priority"),
        ("right_in", 400, 100, "priority"),
        ("right_out", 400, 100, "priority"),
        ("left_in", 0, 300, "priority"),
        ("left_out", 0, 300, "priority"),
        ("top_in", 300, 400, "priority"),
        ("top_out", 300, 400, "priority")
    ]

    for node_id, x, y, ntype in external_nodes:
        node = ET.SubElement(root, "junction")
        node.set("id", node_id)
        node.set("x", str(x))
        node.set("y", str(y))
        node.set("type", ntype)

    # 创建所有边
    edges = horizontal_edges + vertical_edges + external_edges

    for edge_id, from_node, to_node, length in edges:
        edge = ET.SubElement(root, "edge")
        edge.set("id", edge_id)
        edge.set("from", from_node)
        edge.set("to", to_node)
        edge.set("priority", "78")
        edge.set("type", "highway.urban")

        lane = ET.SubElement(edge, "lane")
        lane.set("id", f"{edge_id}_0")
        lane.set("index", "0")
        lane.set("speed", "13.89")
        lane.set("length", str(length))
        lane.set("shape", "")

    # 连接关系（直行）
    connections = [
        ("edge_h0", "edge_v2", 0, 0, 1),  # node0 -> node1 -> node3
        ("edge_v0", "edge_h2", 0, 0, 2),  # node0 -> node2 -> node3
        ("edge_h3", "edge_v1", 0, 0, 1),  # node3 -> node2 -> node0
        ("edge_v3", "edge_h1", 0, 0, 2),  # node3 -> node1 -> node0
    ]

    for from_edge, to_edge, from_lane, to_lane, signal_group in connections:
        conn = ET.SubElement(root, "connection")
        conn.set("from", from_edge)
        conn.set("to", to_edge)
        conn.set("fromLane", str(from_lane))
        conn.set("toLane", str(to_lane))
        conn.set("signalGroup", str(signal_group))

    # 添加交通灯逻辑
    for node_id in ["node0", "node1", "node2", "node3"]:
        tl_logic = ET.SubElement(root, "tlLogic")
        tl_logic.set("id", node_id)
        tl_logic.set("type", "static")
        tl_logic.set("programID", "0")
        tl_logic.set("offset", "0")

        # 相位1: 东西方向绿灯，南北方向红灯
        phase1 = ET.SubElement(tl_logic, "phase")
        phase1.set("duration", "31")
        phase1.set("state", "GGGrrrGGGrrr")

        # 相位2: 黄灯
        phase2 = ET.SubElement(tl_logic, "phase")
        phase2.set("duration", "6")
        phase2.set("state", "yyyrrryyyrrr")

        # 相位3: 南北方向绿灯，东西方向红灯
        phase3 = ET.SubElement(tl_logic, "phase")
        phase3.set("duration", "31")
        phase3.set("state", "rrrGGGrrrGGG")

        # 相位4: 黄灯
        phase4 = ET.SubElement(tl_logic, "phase")
        phase4.set("duration", "6")
        phase4.set("state", "rrryyyrrryyy")

    # 保存文件
    tree = ET.ElementTree(root)
    tree.write("grid_2x2.net.xml", encoding="UTF-8", xml_declaration=True)

    # 美化XML
    import xml.dom.minidom
    dom = xml.dom.minidom.parse("grid_2x2.net.xml")
    pretty_xml = dom.toprettyxml(indent="  ")

    with open("grid_2x2.net.xml", "w") as f:
        f.write(pretty_xml)

    print("✅ 手动创建的网络文件已生成: grid_2x2.net.xml")
    return "grid_2x2.net.xml"


def generate_routes(num_vehicles=20):
    """生成车辆路线"""

    routes_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- 车辆类型定义 -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50" color="1,0,0"/>
    <vType id="bus" accel="1.5" decel="3.0" sigma="0.7" length="12.0" maxSpeed="40" color="0,0,1"/>
    <vType id="truck" accel="1.3" decel="2.5" sigma="0.8" length="16.0" maxSpeed="35" color="0.5,0.5,0.5"/>

    <!-- 路线定义 -->
    <route id="route0" edges="bottom_in node0 node1 right_out"/>
    <route id="route1" edges="left_in node2 node3 top_out"/>
    <route id="route2" edges="bottom_in node0 node2 left_out"/>
    <route id="route3" edges="right_in node1 node3 top_out"/>
    <route id="route4" edges="bottom_in node0 node1 node3 top_out"/>
    <route id="route5" edges="left_in node2 node0 node1 right_out"/>
"""

    # 添加车辆
    depart_time = 0
    for i in range(num_vehicles):
        route_id = random.randint(0, 5)
        vtype_choice = random.random()

        if vtype_choice < 0.7:
            vtype = "car"
        elif vtype_choice < 0.9:
            vtype = "bus"
        else:
            vtype = "truck"

        depart_time += random.randint(1, 5)
        routes_content += f'    <vehicle id="veh{i}" type="{vtype}" route="route{route_id}" depart="{depart_time}" departLane="best"/>\n'

    routes_content += "</routes>"

    with open("grid_2x2.rou.xml", "w") as f:
        f.write(routes_content)

    print(f"✅ 路线文件已生成: grid_2x2.rou.xml (包含{num_vehicles}辆车)")
    return "grid_2x2.rou.xml"


def generate_config(net_file, route_file):
    """生成SUMO配置文件"""

    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="1000"/>
        <step-length value="0.1"/>
    </time>

    <processing>
        <lateral-resolution value="0.25"/>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
        <no-duration-log value="false"/>
    </report>

    <gui_only>
        <gui-settings-file value="grid_2x2.view.xml"/>
        <delay value="50"/>
    </gui_only>

    <output>
        <netstate-dump value="output/grid_2x2.netstate.xml" compressed="true"/>
        <summary-output value="output/grid_2x2.summary.xml"/>
        <tripinfo-output value="output/grid_2x2.tripinfo.xml"/>
    </output>
</configuration>"""

    with open("grid_2x2.sumocfg", "w") as f:
        f.write(config_content)

    print("✅ 配置文件已生成: grid_2x2.sumocfg")
    return "grid_2x2.sumocfg"


def generate_view_settings():
    """生成视图配置文件"""

    view_content = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/viewsettings.xsd">
    <viewport y="200" x="200" zoom="100"/>
    <delay value="50"/>
    <scheme name="real world"/>

    <!-- 背景和道路 -->
    <background color="white"/>
    <show-grid value="false"/>

    <!-- 车辆显示设置 -->
    <vehicle size="1.5" exaggeration="1.0"/>
    <vehicle colorer name="by speed"/>
    <vehicle scalarer name="by speed"/>

    <!-- 车道显示设置 -->
    <lane showLinkDecals="true" width="1.5"/>
    <lane colorer name="by allowed speed"/>
    <lane scalarer name="by allowed speed"/>

    <!-- 交通灯显示 -->
    <junction size="2.0" exaggeration="1.0"/>
    <junction colorer name="by type"/>

    <!-- 其他设置 -->
    <edge name="false"/>
    <edge id="false"/>
    <internal edge="false"/>
</viewsettings>"""

    with open("grid_2x2.view.xml", "w") as f:
        f.write(view_content)

    print("✅ 视图配置文件已生成: grid_2x2.view.xml")
    return "grid_2x2.view.xml"


def create_output_dir():
    """创建输出目录"""
    if not os.path.exists("output"):
        os.makedirs("output")
        print("✅ 创建输出目录: output/")


def main():
    """主函数"""

    print("=" * 60)
    print("SUMO 2x2网格地图生成器")
    print("=" * 60)

    # 创建输出目录
    create_output_dir()

    # 生成网络文件
    net_file = generate_network()

    # 生成路线文件
    route_file = generate_routes(num_vehicles=30)

    # 生成视图设置
    view_file = generate_view_settings()

    # 生成配置文件
    config_file = generate_config(net_file, route_file)

    print("\n" + "=" * 60)
    print("🎉 所有文件生成完成！")
    print("=" * 60)
    print("\n📁 生成的文件列表:")
    print(f"  - {net_file}        (网络文件)")
    print(f"  - {route_file}      (车辆路线文件)")
    print(f"  - {view_file}       (视图设置文件)")
    print(f"  - {config_file}     (主配置文件)")
    print("  - output/            (输出数据目录)")

    print("\n🚀 运行模拟:")
    print("  方法1 (命令行): sumo-gui -c grid_2x2.sumocfg")
    print("  方法2 (无GUI): sumo -c grid_2x2.sumocfg")

    print("\n📊 检查生成的路网:")
    print("  netcheck -s grid_2x2.net.xml")

    print("\n🔧 编辑路网:")
    print("  netedit grid_2x2.net.xml")

    # 询问是否立即运行
    run_now = input("\n是否立即启动SUMO-GUI运行模拟？(y/n): ").strip().lower()
    if run_now == 'y':
        try:
            subprocess.run(["sumo-gui", "-c", "grid_2x2.sumocfg"])
        except FileNotFoundError:
            print("⚠️  sumo-gui未找到，请确保SUMO已正确安装并添加到PATH")
            print("   你可以手动运行: sumo-gui -c grid_2x2.sumocfg")


if __name__ == "__main__":
    main()