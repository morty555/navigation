import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.scene.layout.HBox;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;


class Vertex {
    double x, y;
    int rank,degree;
    int id;
    List<Edge> edges;

    public Vertex(int i, double x, double y) {
        this.id = i;
        this.x = x;
        this.y = y;
        this.rank=1;
        this.degree=0;
        this.edges = new ArrayList<>();
    }

    public List<Vertex> getConnectedVerticesStream() {
        return this.edges.stream()
                .map(e -> e.start == this ? e.end : e.start)
                .distinct()  // 自动去重
                .collect(Collectors.toList());
    }
    // 从顶点列表中随机选择一个顶点并返回。
    public static Vertex getRandomVertex(List<Vertex> vertices) {
        if (vertices == null || vertices.isEmpty()) {
            throw new IllegalArgumentException("Vertex list cannot be null or empty");
        }
        Random random = new Random();
        int index = random.nextInt(vertices.size()); // 生成一个介于[0, vertices.size())的随机索引
        return vertices.get(index); // 返回随机索引对应的顶点
    }

    public double distanceTo(Vertex other) {
        return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
    }

    public void setRank(int i)
    {
        rank=i;
    }
}

class Edge {

    Vertex start, end;
    double length;
    int v; // 车容量
    int n; // 当前车辆数

    public Edge(Vertex start, Vertex end, int v) {
        this.start = start;
        this.end = end;
        this.length = start.distanceTo(end);
        this.v = v;
        this.n = 0; // 初始时道路上的车辆数为0
    }

    // 判断边是否连接给定的两个顶点
    public boolean connects(Vertex v1, Vertex v2) {
        return (start == v1 && end == v2) || (start == v2 && end == v1);
    }

    // 计算路段的通行时间，使用给定的公式
    public double getTrafficTime() {
        double f = (n <= v) ? 1 : (1 + Math.exp((double)n / v));
        return length * f;
    }

    // 更新车辆数
    public void updateTraffic(int cars) {
        this.n += cars;
        if(n<0)
            System.out.println("Edge between " + start.id + " and " + end.id + " has " + n + " cars.");
    }
}

class Graph {
    List<Vertex> vertices;
    List<List<Vertex>> Partition=new ArrayList<List<Vertex>>(600);
    List<List<Edge>> rank_edge=new ArrayList<List<Edge>>(2);

    public Graph() {
        this.vertices = new ArrayList<>();
        for (int p = 0; p < 600; p++) {
            Partition.add(new ArrayList<Vertex>());  // 每个位置初始化一个新的 ArrayList
        }
        for (int p = 0; p < 2; p++) {
            rank_edge.add(new ArrayList<Edge>());  // 每个位置初始化一个新的 ArrayList
        }
    }

    public Map<Integer, Vertex> generateRandomVertices(int N, double maxCoordinateValue) {
        Map<Integer, Vertex> pointsMap = new HashMap<>();
        Random rand = new Random();
        for (int i = 0; i < N; i++) {
            double x = rand.nextDouble() * maxCoordinateValue * 1.5;
            double y = rand.nextDouble() * maxCoordinateValue;
            vertices.add(new Vertex(i, x, y));
            pointsMap.put(i, new Vertex(i, x, y));
            Partition.get((int)(vertices.get(i).x/50)+(int)(vertices.get(i).y/50)*30).add(vertices.get(i));
        }

        return pointsMap;
    }

    public List<Vertex> findNearestVertices(double x,double y, int count) {

        return vertices.stream()
                .sorted(Comparator.comparingDouble(v -> Math.sqrt(Math.pow(v.x - x, 2) + Math.pow(v.y - y, 2))))
                .limit(count)
                .collect(Collectors.toList());
    }

    public List<Edge> getRelatedEdges(List<Vertex> selectedVertices) {
        Set<Vertex> vertexSet = new HashSet<>(selectedVertices);
        return selectedVertices.stream()
                .flatMap(v -> v.edges.stream())
                .filter(e -> vertexSet.contains(e.start) && vertexSet.contains(e.end))
                .collect(Collectors.toList());
    }


    public void generateConnectedGraph(double maxEdgeLength, double connectProbability) {
        List<Edge> allEdges = new ArrayList<>();
        List<Edge> existingEdges = new ArrayList<>();

        for (int i = 0; i < vertices.size(); i++) {
            for (int j = i + 1; j < vertices.size(); j++) {
                Edge edge = new Edge(vertices.get(i), vertices.get(j), (int)(10 +Math.random() * 10));
                if (edge.length <= maxEdgeLength) {
                    allEdges.add(edge);
                }
            }
        }

        allEdges.sort(Comparator.comparingDouble(e -> e.length));

        UnionFind uf = new UnionFind(vertices.size());
        List<Edge> mstEdges = new ArrayList<>();

        for (Edge edge : allEdges) {
            int u = vertices.indexOf(edge.start);
            int v = vertices.indexOf(edge.end);

            if (uf.find(u) != uf.find(v) && !hasIntersection(existingEdges, edge)) {
                uf.union(u, v);
                mstEdges.add(edge);
                existingEdges.add(edge);
                edge.start.edges.add(edge);
                edge.end.edges.add(edge);
            }
        }

        // 添加额外边（避免交叉）
        for (int i = 0; i < vertices.size(); i++) {
            for (int j = i + 1; j < vertices.size(); j++) {
                double distance = vertices.get(i).distanceTo(vertices.get(j));
                if (distance <= maxEdgeLength && Math.random() < connectProbability) {
                    Edge edge = new Edge(vertices.get(i), vertices.get(j), (int)(10 +Math.random() * 10));
                    if (!hasIntersection(existingEdges, edge)) {
                        existingEdges.add(edge);
                        edge.start.edges.add(edge);
                        edge.end.edges.add(edge);
                    }
                }
            }
        }

        for(int i=0;i<Partition.size();i++) {
            for(int j=0;j<Partition.get(i).size();j++) {
                for(int k=0;k<Partition.get(i).get(j).edges.size();k++) {
                    Partition.get(i).get(j).degree+=Partition.get(i).get(j).edges.get(k).v;
                }
            }
        }
        for (int i = 0; i < Math.min(Partition.size(), 600); i++) { // 确保索引不会越界
            List<Vertex> currentPartition = Partition.get(i);
            if (currentPartition != null && !currentPartition.isEmpty()) { // 确保列表非空
                currentPartition.sort(Comparator.comparingInt(vertices -> vertices.degree)); // 排序
                currentPartition.get(0).setRank(3);
                // 确保列表有足够的元素来设置rank
                if (currentPartition.size() > 5) {
                    currentPartition.get(1).setRank(2);
                    currentPartition.get(2).setRank(2);
                    currentPartition.get(3).setRank(2);
                    currentPartition.get(4).setRank(2);
                }
            }
        }

        for (int count = 0; count < 600; count++) {
            if (count > 29) {
                rank_edge.get(0).add(new Edge(Partition.get(count).get(0), Partition.get(count-30).get(0), 0));
            }
            if (count < 570) {
                rank_edge.get(0).add(new Edge(Partition.get(count).get(0), Partition.get(count+30).get(0), 0));
            }
            if (count % 30 != 0) {
                rank_edge.get(0).add(new Edge(Partition.get(count).get(0), Partition.get(count-1).get(0), 0));
            }
            if (count % 30 != 29) {
                rank_edge.get(0).add(new Edge(Partition.get(count).get(0), Partition.get(count+1).get(0), 0));
            }
            for(int k=1;k<Math.min(5,Partition.get(count).size());k++)
            {
                rank_edge.get(1).add(new Edge(Partition.get(count).get(0), Partition.get(count).get(k), 0));
            }
        }
    }

    private boolean hasIntersection(List<Edge> edges, Edge newEdge) {
        for (Edge edge : edges) {
            if (edgesIntersect(edge.start, edge.end, newEdge.start, newEdge.end)) {
                return true;
            }
        }
        return false;
    }

    private boolean edgesIntersect(Vertex a1, Vertex a2, Vertex b1, Vertex b2) {
        return linesIntersect(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y);
    }

    // 几何工具函数（使用线段交叉判断）
    private boolean linesIntersect(double x1, double y1, double x2, double y2,
                                   double x3, double y3, double x4, double y4) {
        double d1 = direction(x3, y3, x4, y4, x1, y1);
        double d2 = direction(x3, y3, x4, y4, x2, y2);
        double d3 = direction(x1, y1, x2, y2, x3, y3);
        double d4 = direction(x1, y1, x2, y2, x4, y4);

        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
                ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }

        return false;
    }

    private double direction(double xi, double yi, double xj, double yj, double xk, double yk) {
        return (xk - xi) * (yj - yi) - (xj - xi) * (yk - yi);
    }

    // 函数：根据两个顶点返回对应的边
    public Edge findEdge(Vertex v1, Vertex v2) {
        if (v1 == null || v2 == null) {
            return null;
        }

        // 遍历第一个顶点的边集合
        for (Edge edge : v1.edges) {
            // 如果边连接了这两个顶点，则返回这个边
            if (edge.connects(v1, v2)) {
                return edge;
            }
        }
        // 如果没有找到对应的边，则返回空
        return null;
    }

    public List<Vertex> getVertices() {
        return vertices;
    }

    public List<Edge> getEdges() {
        return vertices.stream()
                .flatMap(v -> v.edges.stream())
                .collect(Collectors.toList());
    }

    public void scaleMap(double scaleFactor) {
        vertices.forEach(vertex -> {
            vertex.x *= scaleFactor;
            vertex.y *= scaleFactor;
        });
    }

    public List<Vertex> calculateShortestPath(Vertex source, Vertex destination) {
        Map<Vertex, Double> distances = new HashMap<>();
        Map<Vertex, Vertex> predecessors = new HashMap<>();
        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));

        vertices.forEach(vertex -> distances.put(vertex, Double.POSITIVE_INFINITY));
        distances.put(source, 0.0);
        priorityQueue.add(source);

        while (!priorityQueue.isEmpty()) {
            Vertex current = priorityQueue.poll();

            if (current == destination) {
                break;
            }

            current.edges.forEach(edge -> {
                Vertex neighbor = (edge.start == current) ? edge.end : edge.start;
                double newDist = distances.get(current) + edge.getTrafficTime();  // 使用通行时间作为权重

                if (newDist < distances.get(neighbor)) {
                    distances.put(neighbor, newDist);
                    predecessors.put(neighbor, current);
                    priorityQueue.add(neighbor);
                }
            });
        }

        // 构建路径，并确保路径上的顶点之间都有边
        List<Vertex> path = new ArrayList<>();
        for (Vertex at = destination; at != null; at = predecessors.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);

        // 检查路径是否有效（每对相邻顶点之间都有边）
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex v1 = path.get(i);
            Vertex v2 = path.get(i + 1);
            if (findEdge(v1, v2) == null) {
                return new ArrayList<>();  // 如果路径无效，返回空列表
            }
        }

        return path;
    }

    public List<Vertex> calculateBestPath(Vertex source, Vertex destination) {
        Map<Vertex, Double> distances = new HashMap<>();
        Map<Vertex, Vertex> predecessors  = new HashMap<>();
        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));

        vertices.forEach(vertex -> distances.put(vertex, Double.POSITIVE_INFINITY));
        distances.put(source, 0.0);
        priorityQueue.add(source);

        while (!priorityQueue.isEmpty()) {
            Vertex current = priorityQueue.poll();

            if (current == destination) {
                break;
            }

            current.edges.forEach(edge -> {
                Vertex neighbor = (edge.start == current) ? edge.end : edge.start;
                double newDist = distances.get(current)+ edge.length;

                if (newDist < distances.get(neighbor)) {
                    distances.put(neighbor, newDist);
                    predecessors.put(neighbor, current);
                    priorityQueue.add(neighbor);
                }
            });
        }

        // 构建路径，并确保路径上的顶点之间都有边
        List<Vertex> path = new ArrayList<>();
        for (Vertex at = destination; at != null; at = predecessors.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);

        // 检查路径是否有效（每对相邻顶点之间都有边）
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex v1 = path.get(i);
            Vertex v2 = path.get(i + 1);
            if (findEdge(v1, v2) == null) {
                return new ArrayList<>();  // 如果路径无效，返回空列表
            }
        }

        return path;
    }

}

class TimeSlot implements Comparable<TimeSlot>{
    private final double startTime; // 时间槽起始时间（不可变）
    private final List<Car> cars = new ArrayList<>();

    public TimeSlot(double startTime) {
        this.startTime = startTime;
    }

    public double getStartTime() { return startTime; }
    public List<Car> getCars() { return cars; }

    @Override
    public int compareTo(TimeSlot other) {
        return Double.compare(this.startTime, other.startTime);
    }
}

class Car {
    private Vertex currentVertex; // 当前所在点
    private Vertex destinationVertex; // 前往点
    private double travelTime; // 到达下一个地点耗时t

    public Car(Vertex currentVertex, Graph graph) {
        this.currentVertex = currentVertex;
        this.destinationVertex = null;
        this.travelTime = 0;
        setRandomDestination(currentVertex, graph);
        Edge currentEdge = graph.findEdge(currentVertex, destinationVertex);
        currentEdge.updateTraffic(+1);
        travelTime=currentEdge.getTrafficTime();
    }

    public double gettravalTime() {
        return travelTime;
    }

    // 更新车辆状态，包括计时器和路径
    public void update(TrafficSimulation simulation) {
        Graph graph = simulation.getGraph();

        // 记录移动前的原始顶点信息
        Vertex originalStart = currentVertex;
        Vertex originalDestination = destinationVertex;

        // 移动到目的地
        currentVertex = originalDestination;

        // 设置新目的地
        setRandomDestination(currentVertex, graph);

        // 正确查找原路径的边（车辆刚刚离开的边）
        Edge lastEdge = graph.findEdge(originalStart, originalDestination);

        // 查找新路径的边（车辆即将进入的边）
        Edge currentEdge = graph.findEdge(currentVertex, destinationVertex);

        // 更新交通量
        if (lastEdge != null) {
            lastEdge.updateTraffic(-1);  // 正确离开原边
        }
        if (currentEdge != null) {
            currentEdge.updateTraffic(+1);  // 进入新边
            travelTime = currentEdge.getTrafficTime();
        }

    }


    // 设置随机目的地，并计算路径
    public void setRandomDestination(Vertex currentVertex, Graph graph) {
        List<Vertex> possibleDestinations = currentVertex.getConnectedVerticesStream();
        if (possibleDestinations.isEmpty()) return;
        destinationVertex = Vertex.getRandomVertex(possibleDestinations);

    }
}

class UnionFind {
    int[] parent;
    int[] rank;

    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    public void union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
}

class  TrafficSimulation {

    private final PriorityQueue<TimeSlot> timeSlotsQueue;
    private final Map<Double, TimeSlot> timeSlotsMap; // 模拟中的车辆列表
    private Graph graph; // 路网图
    private double simulationTime; // 模拟的总时间（以分钟为单位）
    private double currentTime; // 当前模拟时间（以分钟为单位）
    private double timeStep; // 模拟时间步长（以分钟为单位）
    private double min_timer;//车队列中最小的timer


    public TrafficSimulation(Graph graph, double simulationTime, double timeStep, int cars_num) {
        this.graph = graph;
        this.simulationTime = simulationTime;
        this.timeStep = timeStep;
        this.currentTime = 0;
        this.min_timer = Double.MAX_VALUE;
        this.timeSlotsQueue = new PriorityQueue<>();
        this.timeSlotsMap = new HashMap<>();
        initializeCars(cars_num);
    }

    private void initializeCars(int cars_num) {
        // 初始化车辆，为每辆车设置起始点和目的地
        Random rand = new Random();
        for (int i = 0; i < cars_num; i++) { // 假设我们初始化20000辆车
            System.out.println(i);
            List<Vertex> possiblever = graph.getVertices();
            Vertex start = Vertex.getRandomVertex(possiblever);
            Car car = new Car(start, graph);
            // 计算车辆应属的时间槽起始时间
            double newStartTime = Math.floor((car.gettravalTime()+currentTime) / timeStep) * timeStep;

            // 获取或创建新时间槽
            TimeSlot newSlot = timeSlotsMap.get(newStartTime);
            if (newSlot == null) {
                newSlot = new TimeSlot(newStartTime);
                timeSlotsMap.put(newStartTime, newSlot);
                timeSlotsQueue.add(newSlot);
            }

            // 将车辆加入新时间槽
            newSlot.getCars().add(car);
        }
        min_timer = timeSlotsQueue.peek().getStartTime() ;
    }

    public void startSimulation() {
        new Thread(()->{
            while (currentTime < simulationTime) {
                currentTime += timeStep;
                updateSimulation();
                //模拟休眠一段时间来模拟现实时间流逝，休眠时间取决于时间步长和模拟速度
                try {
                    Thread.sleep((long) (timeStep*3000 ));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    public Graph getGraph() {
        return graph;
    }

    private void updateSimulation() {

        // 循环处理所有满足时间条件的车辆
        if (!timeSlotsQueue.isEmpty() && currentTime > min_timer) {
            // 暂存需要重新调度的车辆
            List<Car> rescheduledCars = new ArrayList<>();
            while (!timeSlotsQueue.isEmpty() && min_timer < currentTime) {
                // 从旧时间槽中移除车辆
                for (Car car:timeSlotsQueue.peek().getCars()) {
                    rescheduledCars.add(car);
                    car.update(this);
                }
                // 如果旧时间槽为空，则清理
                timeSlotsMap.remove(timeSlotsQueue.peek().getStartTime());
                timeSlotsQueue.remove(timeSlotsQueue.peek());
            }
            while (!rescheduledCars.isEmpty() ) {
                Car car=rescheduledCars.remove(0);
                double newStartTime = Math.floor((car.gettravalTime()+currentTime) / timeStep) * timeStep;
                // 获取或创建新时间槽
                TimeSlot newSlot = timeSlotsMap.get(newStartTime);
                if (newSlot == null) {
                    newSlot = new TimeSlot(newStartTime);
                    timeSlotsMap.put(newStartTime, newSlot);
                    timeSlotsQueue.add(newSlot);
                }
                // 将车辆加入新时间槽
                newSlot.getCars().add(car);
            }
            min_timer=timeSlotsQueue.peek().getStartTime();
        }
    }
}


public class data extends Application {

    private double scaleFactor = 4.0;
    private static final double SCALE_SPEED = 0.05;
    private static final double MIN_SCALE = 1;
    private static final double MAX_SCALE = 5.0;

    private double translateX = 0, translateY = 0;  // 用于平移地图的位置
    private double mousePressedX, mousePressedY;

    AtomicInteger judgeshortest = new AtomicInteger(1);
    AtomicInteger judgebest = new AtomicInteger(1);

    private List<Vertex> shortestPath = new ArrayList<>();
    AtomicReference<List<Vertex>> nearestVertices = new AtomicReference<>(new ArrayList<>());
    ;
    AtomicReference<List<Edge>> relatedEdges = new AtomicReference<>(new ArrayList<>());

    @Override
    public void start(Stage primaryStage) {

        int N = 10000;
        double maxCoordinateValue = 1000;
        double maxEdgeLength = 100;
        double connectProbability = 0.2;

        Graph graph = new Graph();
        Map<Integer, Vertex> vertexMap = graph.generateRandomVertices(N, maxCoordinateValue);
        graph.generateConnectedGraph(maxEdgeLength, connectProbability);

        if (graph.getVertices().isEmpty()) {
            System.err.println("Error: Graph has no vertices.");
            return;
        }
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }

        Canvas canvas = new Canvas(1800, 1000);
//        primaryStage.setWidth(1550);
//        primaryStage.setHeight(1100);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setLineWidth(2);

        //如果 graph.getVertices() 为空或 size() < 2，这会导致 IndexOutOfBoundsException，但也可能在某些情况下变成 NullPointerException。
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }
        //工具栏

        //find 100 nearest vertex
        ToolBar toolBar = new ToolBar();
        TextField xInput = new TextField();
        TextField yInput = new TextField();
        TextField startPoint = new TextField("0");
        TextField endPoint = new TextField("1");
        TextField pointInput = new TextField();
        AtomicReference<Vertex> source = new AtomicReference<>(graph.getVertices().get(getTextFieldPointID(startPoint)));
        AtomicReference<Vertex> destination = new AtomicReference<>(graph.getVertices().get(getTextFieldPointID(endPoint)));

        Button searchButton = new Button("查找最近100个顶点");
//        toolBar.getItems().addAll(new Label("pointID:"), pointInput, searchButton);
        toolBar.getItems().addAll(new Label("X:"), xInput, new Label("Y:"), yInput, searchButton);

        //处理逻辑

        // TODO: 2025/3/20 输入点数据时才会调用drawmap展示最近100个顶点高亮，在进行其他操作如放大缩小等再次调用drawmap函数时才会展示高亮，因此高亮只会存在一瞬间，进行其他操作之后才会继续产生高亮
        // TODO: 2025/3/22 打算修改成需要产生最短路径和100个顶点高亮时生成那一时刻的静态页面并new一个新的GUI  因为动态车流显示需要时刻绘画边长颜色，如果想要最短路径和顶点高亮一直存在不现实。
        searchButton.setOnAction(e -> {
            try {
                double x = Double.parseDouble(xInput.getText());
                double y = Double.parseDouble(yInput.getText());
//                int id = Integer.parseInt(pointInput.getText());
                // 查找最近100个顶点
                nearestVertices.set(graph.findNearestVertices(x,y, 100));
                relatedEdges.set(graph.getRelatedEdges(nearestVertices.get()));

                // 重新绘制地图
                scaleFactor = 5;
                translateX = 180-x;
                translateY = 100-y;
                redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());

            } catch (NumberFormatException ex) {
                System.out.println("请输入有效的数字");
            }
        });

        //放大缩小
        Button zoomInButton = new Button("放大");
        Button zoomOutButton = new Button("缩小");
        Button resetViewButton = new Button("重置视图");
        ComboBox<String> pathOptions1 = new ComboBox<>();
        ComboBox<String> pathOptions2 = new ComboBox<>();
        //起点终点选取

        pathOptions1.getItems().addAll("显示最优路径", "隐藏最优路径");
        pathOptions1.setValue("显示最优路径");

        pathOptions2.getItems().addAll("显示最短路径", "隐藏最短路径");
        pathOptions2.setValue("显示最短路径");

        ToggleGroup toggleGroup = new ToggleGroup();
        ToggleButton setStartPointButton = new ToggleButton("选择起点:");
        setStartPointButton.setToggleGroup(toggleGroup);
        ToggleButton setEndPointButton = new ToggleButton("选择终点:");
        setEndPointButton.setToggleGroup(toggleGroup);



        toolBar.getItems().addAll(zoomInButton, zoomOutButton, resetViewButton, pathOptions1,pathOptions2);
        toolBar.getItems().addAll(setStartPointButton, startPoint, setEndPointButton, endPoint);

        // 添加事件处理
        zoomInButton.setOnAction(e -> {
            scaleFactor = Math.min(MAX_SCALE, scaleFactor + SCALE_SPEED);
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        zoomOutButton.setOnAction(e -> {
            scaleFactor = Math.max(MIN_SCALE, scaleFactor - SCALE_SPEED);
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        resetViewButton.setOnAction(e -> {
            scaleFactor = 4.0;
            translateX = 0;
            translateY = 0;
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });


        pathOptions1.setOnAction(e -> {
            if (pathOptions1.getValue().equals("显示最优路径")) {
                judgeshortest.set(1);
                System.out.println(pathOptions1.getValue());
                redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());

            } else {
                judgeshortest.set(0);
                System.out.println(pathOptions1.getValue());
                redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get()); // 只重绘地图，不显示路径
            }
        });

        pathOptions2.setOnAction(e -> {
            if (pathOptions2.getValue().equals("显示最短路径")) {
                judgebest.set(1);
                System.out.println(pathOptions2.getValue());
                redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());

            } else {
                judgebest.set(0);
                System.out.println(pathOptions2.getValue());
                redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get()); // 只重绘地图，不显示路径
            }
        });

        // 绘制地图

        drawMap(scaleFactor,gc, graph, source.get(), destination.get(), judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());

        double simulationTime=100000;
        double timeStep=1;
        int cars_num=200000;
        TrafficSimulation trafficSimulation = new TrafficSimulation(graph, simulationTime, timeStep,cars_num);
        trafficSimulation.startSimulation();
        new Thread(()->{
            while (true){

                updateMap(gc, trafficSimulation.getGraph(),source.get(),destination.get(),judgebest, judgeshortest,nearestVertices.get(), relatedEdges.get());

                System.out.println("结束一次车流更新");

                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            // 更新车流模拟

        }).start();

        // 鼠标事件
        canvas.setOnMousePressed(event -> {
            mousePressedX = event.getSceneX();
            mousePressedY = event.getSceneY();

            if(setStartPointButton.isSelected())
            {
                double toolbarHeight = toolBar.getHeight();
                int pointID = getPointIDSelected(mousePressedX, mousePressedY, toolbarHeight, gc, graph);
                if(pointID != -1) {
                    startPoint.setText(String.valueOf(pointID));
                    source.set(graph.getVertices().get(getTextFieldPointID(startPoint)));
                }
            }

            if(setEndPointButton.isSelected())
            {
                double toolbarHeight = toolBar.getHeight();
                int pointID = getPointIDSelected(mousePressedX, mousePressedY, toolbarHeight, gc, graph);
                if(pointID != -1) {
                    endPoint.setText(String.valueOf(pointID));
                    destination.set(graph.getVertices().get(getTextFieldPointID(endPoint)));
                }
            }
        });

        canvas.setOnMouseDragged(event -> {
            double deltaX = event.getSceneX() - mousePressedX;
            double deltaY = event.getSceneY() - mousePressedY;

            translateX += deltaX;
            translateY += deltaY;

            // 更新鼠标位置
            mousePressedX = event.getSceneX();
            mousePressedY = event.getSceneY();

            // 重绘地图
            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
            drawMap(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
            //displayShortestPath(gc, graph, source, destination);
        });

        // 缩放监听事件
        canvas.setOnScroll((ScrollEvent event) -> {
            scaleFactor += (event.getDeltaY() > 0) ? SCALE_SPEED : -SCALE_SPEED;
            scaleFactor = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scaleFactor));


            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

            drawMap(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
            //displayShortestPath(gc, graph, source, destination);
        });


        VBox root = new VBox(toolBar, canvas);

        // 创建并显示场景
        Scene scene = new Scene(root, 1800, 1000);
        primaryStage.setTitle("Graph Visualization");
        primaryStage.setScene(scene);
        primaryStage.show();

    }

    private int getPointIDSelected(double mousePressedX, double mousePressedY, double toolbarHeight, GraphicsContext gc, Graph graph)
    {
        double clickRadius = 5;

        double drawX = mousePressedX - clickRadius;  // 修正 X 坐标
        double drawY = mousePressedY - toolbarHeight - clickRadius;  // 修正 Y 坐标

//        gc.setFill(Color.YELLOW);
//        gc.fillOval(drawX, drawY, clickRadius * 2, clickRadius * 2);

        // 遍历所有顶点，检查是否有顶点被圆覆盖
        for (Vertex vertex : graph.getVertices()) {
            if(vertex.rank < printrank) continue;

            double distance = Math.sqrt(Math.pow(drawX - (vertex.x + translateX) * scaleFactor, 2)
                    + Math.pow(drawY - (vertex.y + translateY) * scaleFactor, 2));

            if (distance <= clickRadius * 2) {
                gc.setFill(Color.BLUE);
                gc.fillOval((vertex.x + translateX) * scaleFactor - 5, (vertex.y + translateY) * scaleFactor - 5, 10, 10);
                return vertex.id;
            }
        }
        return -1;
    }

    private int getTextFieldPointID(TextField textField)
    {
        String text = textField.getText().trim();
        if (text.matches("\\d+")) {
            return Integer.parseInt(text);
        }
        return 0;
    }

    private void reflashSource()
    {

    }

    private void redraw(double scaleFactor,GraphicsContext gc, Graph graph, Vertex source, Vertex destination,AtomicInteger judgeBest, AtomicInteger judgeshortest, List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
        drawMap(scaleFactor,gc, graph, source, destination,  judgeBest,judgeshortest, highlightVertices, highlightEdges);
        // System.out.println(highlightEdges);

    }

    int printrank;
    private void drawMap(double scaleFactor,GraphicsContext gc, Graph graph, Vertex source, Vertex destination,AtomicInteger judgeBest, AtomicInteger judgeshortest, List<Vertex> highlightVertices, List<Edge> highlightEdges) {

        if(scaleFactor>=3.5&&scaleFactor<=5)printrank=1;
        else if(scaleFactor>=1&&scaleFactor<=2)printrank=3;
        else printrank=2;
        if(printrank==1) {
            graph.getEdges().forEach(edge -> {
                Color edgeColor = getEdgeColor(edge);
                gc.setStroke(edgeColor);
                gc.strokeLine(
                        (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                        (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
                );
            });
        } else if (printrank==2) {
            graph.rank_edge.get(1).forEach(edge->{
                gc.setStroke(Color.GRAY);
                gc.strokeLine(
                        (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                        (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
                );
            });
            graph.rank_edge.get(0).forEach(edge->{
                gc.setStroke(Color.GRAY);
                gc.strokeLine(
                        (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                        (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
                );
            });
        } else {
            graph.rank_edge.get(0).forEach(edge->{
                gc.setStroke(Color.BLACK);
                gc.strokeLine(
                        (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                        (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
                );
            });
        }

        for(List<Vertex>VList :graph.Partition)
        {

            for(Vertex v:VList)
            {
                Color C;
                if(v.rank==3){
                    C=Color.BLACK;
                }else C=Color.RED;
                if(v.rank>=printrank) {
                    gc.setFill(C);
                    gc.fillOval(
                            (v.x + translateX) * scaleFactor - 5,
                            (v.y + translateY) * scaleFactor - 5,
                            10, 10
                    );
                }
            }
        }

        if (judgeBest.get() == 1) {
            displayBestPath(gc, graph, source, destination);
            //System.out.println(judgeshortest.get());
        }

        if (judgeshortest.get() == 1) {
            displayShortestPath(gc, graph, source, destination);
            //System.out.println(judgeshortest.get());
        }
        displayNearVertex(gc, graph, source, destination, highlightVertices, highlightEdges);
    }

//private void drawMap(double scaleFactor,GraphicsContext gc, Graph graph, Vertex source, Vertex destination,AtomicInteger judgeshortest,List<Vertex> highlightVertices, List<Edge> highlightEdges) {
//
//    //初始化点和边
//    graph.getEdges().forEach(edge -> {
//        Color edgeColor = getEdgeColor(edge);
//        gc.setStroke(edgeColor);
//
//        gc.strokeLine(
//                (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
//                (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
//        );
//    });
//
//    graph.getVertices().forEach(vertex -> {
//        gc.setFill(Color.RED);
//        gc.fillOval(
//                (vertex.x + translateX) * scaleFactor - 5,
//                (vertex.y + translateY) * scaleFactor - 5,
//                10, 10
//        );
//    });
//
//    if(judgeshortest.get() == 1){
//        displayShortestPath(gc, graph, source, destination);
//        //System.out.println(judgeshortest.get());
//    }
//    // 高亮最近的100个顶点
//       /* gc.setFill(Color.BLUE);
//        highlightVertices.forEach(vertex -> {
//            gc.fillOval(
//                    (vertex.x + translateX) * scaleFactor - 5,
//                    (vertex.y + translateY) * scaleFactor - 5,
//                    10, 10
//            );
//        });
//
//        // 高亮相关的边
//        gc.setStroke(Color.ORANGE);
//        highlightEdges.forEach(edge -> {
//            gc.strokeLine(
//                    (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
//                    (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
//            );
//        });*/
//
//    displayNearVertex(gc,graph,source,destination,highlightVertices,highlightEdges);
//}

    //展示最近一百个点
    private void displayNearVertex(GraphicsContext gc, Graph graph, Vertex source, Vertex destination, List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        // 高亮最近的100个顶点
        gc.setFill(Color.BLUE);
        highlightVertices.forEach(vertex -> {
            gc.fillOval(
                    (vertex.x + translateX) * scaleFactor - 5,
                    (vertex.y + translateY) * scaleFactor - 5,
                    10, 10
            );
        });

        // 高亮相关的边
        gc.setStroke(Color.ORANGE);
        highlightEdges.forEach(edge -> {
            gc.strokeLine(
                    (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                    (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
            );
        });
        // System.out.println(highlightEdges);
    }

    private Color getEdgeColor(Edge edge) {
        double traffic = edge.getTrafficTime();
        if (edge.n<0.6*edge.v) {
            return Color.GREEN; // 轻度拥堵
        } else if (0.6*edge.v<=edge.n&&edge.n<=edge.v) {
            return Color.YELLOW; // 中度拥堵
        } else {
            return Color.RED; // 高度拥堵
        }
    }

    private void updateMap(GraphicsContext gc, Graph graph,Vertex source,Vertex destination,AtomicInteger judgebest,AtomicInteger judgeshortest,List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        // 更新并绘制地图
        Platform.runLater(() -> {
            gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
            drawMap(scaleFactor,gc, graph, source, destination,judgebest, judgeshortest, highlightVertices, highlightEdges);
        });

        graph.getEdges().forEach(edge -> {
            System.out.printf("Edge %d-%d: n=%d, v=%d, trafficTime=%.1f%n",
                    edge.start.id, edge.end.id, edge.n, edge.v, edge.getTrafficTime());
        });

    }

    // TODO: 2025/3/20  展示最优路径，要结合车流量和最短路径综合考虑


    // TODO: 2025/3/20  地图缩放功能只展示重要点的功能。method：可能需要在每个区域set一个特殊点。可能生成连通图的方式需要优化。

    // TODO: 2025/3/21  随着缩放图片或者放大窗口，地图能随着自定义布局。

    // 计算并显示最短路径
    static int outputtimes = 1;

    private void displayBestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

        List<Vertex> path = graph.calculateBestPath(source, destination);

        gc.setStroke(Color.BLUE);
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex start = path.get(i);
            Vertex end = path.get(i + 1);
            gc.strokeLine(
                    (start.x + translateX) * scaleFactor, (start.y + translateY) * scaleFactor,
                    (end.x + translateX) * scaleFactor, (end.y + translateY) * scaleFactor);
            if (outputtimes >= 1) {
                System.out.println("start point: (" + start.x * scaleFactor + "," + start.y * scaleFactor + ")" +
                        "end point: (" + end.x * scaleFactor + "," + end.y * scaleFactor + ")");
            }
        }
        outputtimes = 0;
    }
    private void displayShortestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

        List<Vertex> path = graph.calculateShortestPath(source, destination);

        gc.setStroke(Color.BLUE);
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex start = path.get(i);
            Vertex end = path.get(i + 1);
            gc.strokeLine(
                    (start.x + translateX) * scaleFactor, (start.y + translateY) * scaleFactor,
                    (end.x + translateX) * scaleFactor, (end.y + translateY) * scaleFactor);
            if (outputtimes >= 1) {
                System.out.println("start point: (" + start.x * scaleFactor + "," + start.y * scaleFactor + ")" +
                        "end point: (" + end.x * scaleFactor + "," + end.y * scaleFactor + ")");
            }
        }
        outputtimes = 0;
    }

}













