import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.ScrollEvent;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.util.*;
import java.util.stream.Collectors;


class Vertex {
    double x, y;
    List<Edge> edges;

    public Vertex(double x, double y) {
        this.x = x;
        this.y = y;
        this.edges = new ArrayList<>();
    }

    public double distanceTo(Vertex other) {
        return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
    }
}

class Edge {
    Vertex start, end;
    double length;
    double v; // 车容量
    double n; // 当前车辆数

    public Edge(Vertex start, Vertex end, double v) {
        this.start = start;
        this.end = end;
        this.length = start.distanceTo(end);
        this.v = v;
        this.n = 0; // 初始时道路上的车辆数为0
    }

    // 计算路段的通行时间，使用给定的公式
    public double getTrafficTime() {
        double f = (n <= v) ? 1 : (1 + Math.exp(n / v));
        return length * f;
    }

    // 更新车辆数
    public void updateTraffic(double cars) {
        this.n += cars;
    }
}

class Graph {
    List<Vertex> vertices;

    public Graph() {
        this.vertices = new ArrayList<>();
    }

    public void generateRandomVertices(int N, double maxCoordinateValue) {
        Random rand = new Random();
        for (int i = 0; i < N; i++) {
            double x = rand.nextDouble() * maxCoordinateValue;
            double y = rand.nextDouble() * maxCoordinateValue;
            vertices.add(new Vertex(x, y));
        }
    }

    public void generateConnectedGraph(double maxEdgeLength) {
        List<Edge> allEdges = new ArrayList<>();
        for (int i = 0; i < vertices.size(); i++) {
            for (int j = i + 1; j < vertices.size(); j++) {
                Edge edge = new Edge(vertices.get(i), vertices.get(j), 10 + Math.random() * 10); // 随机车容量
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

            if (uf.find(u) != uf.find(v)) {
                uf.union(u, v);
                mstEdges.add(edge);
            }
        }

        mstEdges.forEach(edge -> {
            edge.start.edges.add(edge);
            edge.end.edges.add(edge);
        });
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
                double newDist = distances.get(current) + edge.length;

                if (newDist < distances.get(neighbor)) {
                    distances.put(neighbor, newDist);
                    predecessors.put(neighbor, current);
                    priorityQueue.add(neighbor);
                }
            });
        }

        List<Vertex> path = new ArrayList<>();
        for (Vertex at = destination; at != null; at = predecessors.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);
        return path;
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

public class data extends Application {

    private double scaleFactor = 1.0;
    private static final double SCALE_SPEED = 0.05;
    private static final double MIN_SCALE = 0.1;
    private static final double MAX_SCALE = 5.0;

    private double translateX = 0, translateY = 0;  // 用于平移地图的位置
    private double mousePressedX, mousePressedY;

    private List<Vertex> shortestPath = new ArrayList<>();

    @Override
    public void start(Stage primaryStage) {
        int N = 1000;
        double maxCoordinateValue = 800;
        double maxEdgeLength = 100;

        Graph graph = new Graph();
        graph.generateRandomVertices(N, maxCoordinateValue);
        graph.generateConnectedGraph(maxEdgeLength);

        // 创建 Canvas 绘制图形
        Group root = new Group();
        Canvas canvas = new Canvas(maxCoordinateValue, maxCoordinateValue);
        root.getChildren().add(canvas);

        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setLineWidth(2);

        // 绘制地图
        drawMap(gc, graph);

        // 最短路径
        Vertex source = graph.getVertices().get(0);
        Vertex destination = graph.getVertices().get(1);
        displayShortestPath(gc, graph, source, destination);

        // 更新车流模拟
        simulateTraffic(gc, graph);

        // 鼠标拖动事件
        canvas.setOnMousePressed(event -> {
            mousePressedX = event.getSceneX();
            mousePressedY = event.getSceneY();
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
            drawMap(gc, graph);
            displayShortestPath(gc, graph, source, destination);
        });

        // 缩放监听事件
        canvas.setOnScroll((ScrollEvent event) -> {
            scaleFactor += (event.getDeltaY() > 0) ? SCALE_SPEED : -SCALE_SPEED;
            scaleFactor = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scaleFactor));

            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

            drawMap(gc, graph);
            displayShortestPath(gc, graph, source, destination);
        });

        // 创建并显示场景
        Scene scene = new Scene(root, maxCoordinateValue, maxCoordinateValue);
        primaryStage.setTitle("Graph Visualization");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void drawMap(GraphicsContext gc, Graph graph) {
        graph.getEdges().forEach(edge -> {
            Color edgeColor = getEdgeColor(edge);
            gc.setStroke(edgeColor);

            gc.strokeLine(
                    (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                    (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
            );
        });

        graph.getVertices().forEach(vertex -> {
            gc.setFill(Color.RED);
            gc.fillOval(
                    (vertex.x + translateX) * scaleFactor - 5,
                    (vertex.y + translateY) * scaleFactor - 5,
                    10, 10
            );
        });
    }

    private Color getEdgeColor(Edge edge) {
        double traffic = edge.getTrafficTime();
        if (traffic < 50) {
            return Color.GREEN; // 轻度拥堵
        } else if (traffic < 100) {
            return Color.YELLOW; // 中度拥堵
        } else {
            return Color.RED; // 高度拥堵
        }
    }

    private void simulateTraffic(GraphicsContext gc, Graph graph) {
        Random rand = new Random();

        // 模拟车流
        new Thread(() -> {
            while (true) {
                double judge = 2 * Math.random() - 1;

                try {
                    Thread.sleep(1000); // 每秒更新一次车流
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                // 随机增加或减少一些车辆
                graph.getEdges().forEach(edge -> {
                    edge.updateTraffic(rand.nextInt(10) * judge*10);

                });
                System.out.println(judge);

                // 更新并绘制地图
                gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
                drawMap(gc, graph);
            }
        }).start();
    }

    // 计算并显示最短路径
    static int outputtimes = 1;
    private void displayShortestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

        List<Vertex> path = graph.calculateShortestPath(source, destination);

        gc.setStroke(Color.BLUE);
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex start = path.get(i);
            Vertex end = path.get(i + 1);
            gc.strokeLine(
                    (start.x + translateX) * scaleFactor, (start.y + translateY) * scaleFactor,
                    (end.x + translateX) * scaleFactor, (end.y + translateY) * scaleFactor);
            if(outputtimes>=1){
                System.out.println("start point: (" + start.x * scaleFactor+","+start.y * scaleFactor+")"+
                        "end point: (" + end.x * scaleFactor+","+end.y * scaleFactor+")");}
            }
          outputtimes=0;
        }



    public static void main(String[] args) {
        launch(args);
    }
}













