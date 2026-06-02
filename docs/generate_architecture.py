# docs/generate_architecture.py
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.programming.framework import Flask

graph_attr = {
    "fontsize": "14",
    "bgcolor": "#FFFFFF",        # light background for high contrast
    "fontcolor": "black",
    "pad": "0.5",
    "splines": "curved",
    "nodesep": "0.6",
    "ranksep": "0.8",
}

node_attr = {
    "fontsize": "12",
    "fontcolor": "black",
}

with Diagram(
    "BeamML Project Architecture",
    filename="docs/images/architecture",   # saves as architecture.png
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
):
    user = Users("Civil Engineer\n(Sagar Kore)")


    with Cluster("Web Dashboard"):
        ui = Server("HTML/JS UI\n(Chart.js & SVG)")
        api = Flask("Flask Backend\n(app.py)")

    with Cluster("ML Inference Engine"):
        pinn = Python("BeamPINN Model\n(Physics-Informed)")
        net = Python("BeamNet Model\n(Standard MLP)")

    user >> Edge(color="black", label="interacts") >> ui
    ui >> Edge(color="black", label="requests API") >> api
    api >> Edge(color="#006400", label="loads weights") >> pinn
    api >> Edge(color="#8B0000", label="loads weights") >> net


print("✅ docs/images/architecture.png generated")
