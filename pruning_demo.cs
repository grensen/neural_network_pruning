// C# 11 required
using System.Diagnostics;

PrintUserInfo();

string path = @"C:\pruning\";
string filePath = path + @"pruning_net.txt";

AutoData d= new(path); // get data

// define neural network 
int[] net        = { 784, 100, 100, 10 };
var LEARNINGRATE = 0.0005f;
var MOMENTUM     = 0.67f;
var EPOCHS       = 50;
var BATCHSIZE    = 800;
var FACTOR       = 0.99f;
var PRUNING      = 0.01f;
var SEED         = 1337;
var PARALLEL     = true;

PrintHyperparameters();

var pruning = PruningNetorkMemoryAllocation(net);

PruningWeightInit(net, pruning.positions, pruning.weights, SEED);

RunTrainingWithPruning(PARALLEL, d, net, pruning.positions, pruning.weights, 60000, 
    LEARNINGRATE, MOMENTUM, PRUNING, FACTOR, EPOCHS, BATCHSIZE);

RunTestWithPruning(PARALLEL, d, net, pruning.positions, pruning.weights, 10000);

Console.WriteLine($"\nFinal weights size = {GetWeightsSize(pruning.weights)}");
Console.WriteLine($"Final positions size = {GetWeightsSize(pruning.positions)}");

Console.WriteLine($"\nSave pruned network to: " + filePath);
SavePruningNetToFile(filePath, net, pruning.positions, pruning.weights);

Console.WriteLine($"\nLoad pruned network from: " + filePath);
var net2 = LoadPruningNetFromFile(filePath);
RunTestWithPruning(PARALLEL, d, net2.net, net2.positions, net2.weights, 10000);

Console.WriteLine($"\nLoaded weights size = {GetWeightsSize(net2.weights)}");
Console.WriteLine("\nEnd demo");

Console.ReadLine();
//+------------------------------------------------------------------------+

static void RunTrainingWithPruning(bool multiCore, AutoData d, 
    int[] net, int[][] positions, float[][] weights,  
    int len, float lr, float mom, float threshold, float FACTOR, int EPOCHS, int BATCHSIZE)
{

    Stopwatch stopwatch = Stopwatch.StartNew();

    Console.WriteLine($"\nTraining Progress{(multiCore ? " - Parallel" : "")}:");
    
    float[][] deltas = new float[weights.Length][];
    for (int i = 0; i < weights.Length; i++)
        deltas[i] = new float[weights[0].Length];

    int sourceSize = GetWeightsSize(weights);
    int networkSize = net.Sum();
    int inputSize = net[0];

    // run training
    for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        bool[] c = new bool[B * BATCHSIZE]; // for proper parallel count

        for (int b = 0; b < B; b++)
        {
            if (multiCore)
                Parallel.For(b * BATCHSIZE, (b + 1) * BATCHSIZE, x =>
                {
                    c[x] = PruningTrain(d.samplesTrainingF.AsSpan().Slice(x * inputSize, inputSize), d.labelsTraining[x], net, positions, weights, deltas, networkSize);
                });
            else
                for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                    c[x] = PruningTrain(d.samplesTrainingF.AsSpan().Slice(x * inputSize, inputSize), d.labelsTraining[x], net, positions, weights, deltas, networkSize);

            // sgd mini batch
            PruningSGD(positions, weights, deltas, lr, mom);
        }

        // pruning
        if (epoch == EPOCHS / 2)
            PruningUnstructured(positions, weights, deltas, threshold);

        if ((epoch + 1) % 5 == 0)
        {
            int wSize = GetWeightsSize(weights);
            PrintInfo($"Epoch = {1 + epoch,3} | Weights = {((double)wSize / sourceSize) * 100,6:F2}% |", c.Count(n => n), B * BATCHSIZE, stopwatch);
        }
    }
}
static void RunTestWithPruning(bool multiCore, AutoData d, 
    int[] net, int[][] positions, float[][] weights, int len)
{
    // Console.WriteLine($"\nTest Results{(multiCore ? " - Parallel" : "")}:");
   // Console.WriteLine($"\n");
    bool[] c = new bool[len]; // for proper parallel count

    Stopwatch stopwatch = Stopwatch.StartNew();

    int networkSize = net.Sum();
    int inputSize = net[0];

    if (multiCore)
        Parallel.For(0, len, x =>
        {
            c[x] = PruningTest(d.samplesTestF.AsSpan().Slice(x * inputSize, inputSize), d.labelsTest[x], net, positions, weights, networkSize);
        });
    else // single core
        for (int x = 0; x < len; x++)
            c[x] = PruningTest(d.samplesTestF.AsSpan().Slice(x * inputSize, inputSize), d.labelsTest[x], net, positions, weights, networkSize);

    stopwatch.Stop();

    PrintInfo("\nTest", c.Count(n => n), len, stopwatch, true);
}
static bool PruningTrain(Span<float> sample, int target, 
    int[] net, int[][] positions, float[][] weights, float[][] deltas, int networkSize)
{    
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    PruningFeedForward(neurons, net, positions, weights);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // bp
    Softmax(outs);
    if (outs[target] < 0.99f)
        PruningBackprop(neurons, net, positions, weights, deltas, target);

    return target == prediction;
}
static bool PruningTest(Span<float> sample, int target, 
    int[] net, int[][] positions, float[][] weights, int networkSize)
{    
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    PruningFeedForward(neurons, net, positions, weights);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);
    return target == prediction;
}

static int GetWeightsSize<T>(T[][] array)
{
    int size = 0;
    for (int i = 0; i < array.Length; i++)
        size += array[i].Length;
    return size;
}
// 1. memory allocation
static (int[][] positions, float[][] weights) PruningNetorkMemoryAllocation(int[] net)
{
    int netwrokSize = net.Sum();
    int[][] positions = new int[netwrokSize - net[^1]][];
    float[][] weights = new float[netwrokSize - net[^1]][];

    for (int i = 0, j = 0; i < net.Length - 1; i++)
    {
        for (int l = 0; l < net[i]; l++)
        {
            positions[j + l] = new int[net[i + 1]];
            weights[j + l] = new float[net[i + 1]];
        }
        j += net[i];
    }
    return (positions, weights);
}
// 2. weights init
static void PruningWeightInit(int[] net, int[][] positions, float[][] weights, int seed)
{
    Random rnd = new Random(seed);
    for (int j = 0, k = net[0], i = 0; i < net.Length - 1; i++)
    {
        float sd = MathF.Sqrt(6.0f / (net[i] + net[i + 1]));
        for (int l = 0; l < net[i]; l++)
            for (int r = 0; r < net[i + 1]; r++)
            {
                positions[j + l][r] = k + r;
                weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
            }
        j += net[i]; k += net[i + 1];
    }
}
// 3. feed forward
static void PruningFeedForward(float[] neurons, int[] net, int[][] positions, float[][] weights)
{
    for (int j = 0, i = 0; i < net.Length - 1; i++)
    {
        for (int l = 0; l < net[i]; l++)
        {
            var n = neurons[j + l];
            if (n <= 0) continue;
            for (int r = 0; r < weights[j + l].Length; r++)
                neurons[positions[j + l][r]] += weights[j + l][r] * n;
        }
        j += net[i];
    }
}
// 3.1. prediction
static int Argmax(Span<float> neurons)
{
    int id = 0;
    for (int i = 1; i < neurons.Length; i++)
        if (neurons[i] > neurons[id])
            id = i;
    return id; // prediction
}
// 4. probabilities
static void Softmax(Span<float> neurons)
{
    float scale = 0;
    for (int n = 0; n < neurons.Length; n++)
        scale += neurons[n] = MathF.Exp(neurons[n]); // activation then sum up
    for (int n = 0; n < neurons.Length; n++)
        neurons[n] /= scale; // probabilities
}
// 4.1. backpropagation
static void PruningBackprop(float[] neurons, int[] net, int[][] positions, float[][] weight, float[][] delta, int target)
{
    Span<float> gradients = new float[neurons.Length]; // right output side

    for (int r = neurons.Length - net[^1], p = 0; r < neurons.Length; r++, p++)
        gradients[r] = target == p ? 1 - neurons[r] : -neurons[r];

    for (int j = neurons.Length - net[^1], i = net.Length - 2; i >= 0; i--)
    {
        j -= net[i];
        for (int l = 0; l < net[i]; l++)
        {
            var jl = j + l;
            var n = neurons[jl];
            if (n <= 0) continue;
            float sum = 0;
            for (int r = 0; r < weight[jl].Length; r++)
            {
                float gra = gradients[positions[jl][r]];
                sum += weight[jl][r] * gra;
                delta[jl][r] += n * gra;
            }
            gradients[jl] = sum;
        }
    }
}
// 5. weight update
static void PruningSGD(int[][] positions, float[][] weights, float[][] delta, float lr, float mom)
{
    for (int i = 0; i < positions.Length; i++)
        for (int j = 0; j < positions[i].Length; j++)
        {
            weights[i][j] += delta[i][j] * lr;
            delta[i][j] *= mom;
        }
}
// 6. weights pruning 
static void PruningUnstructured(int[][] positions, float[][] weights, float[][] delta, float threshold)
{
    for (int i = 0; i < positions.Length; i++)
    {
        for (int j = 0; j < positions[i].Length; j++)
        {
            if (MathF.Pow(weights[i][j], 2) < threshold)
            {
                // Remove weight from all arrays
                for (int k = j; k < positions[i].Length - 1; k++)
                {
                    positions[i][k] = positions[i][k + 1];
                    weights[i][k] = weights[i][k + 1];
                    delta[i][k] = delta[i][k + 1];
                }

                // Reduce the array size by one
                Array.Resize(ref positions[i], positions[i].Length - 1);
                Array.Resize(ref weights[i], weights[i].Length - 1);
                Array.Resize(ref delta[i], delta[i].Length - 1);

                // Decrement j to re-check the current index
                j--;
            }
        }
    }
}
// 7. save network weights and its positions  
static void SavePruningNetToFile(string fileName, int[] net, int[][] positions, float[][] weights)
{
    using (StreamWriter writer = new StreamWriter(fileName))
    {
        // Write the network architecture (net array)
        writer.WriteLine(string.Join(",", net));

        // Write the positions array
        foreach (var positionRow in positions)
        {
            writer.WriteLine(string.Join(",", positionRow));
        }

        // Write the weights array
        foreach (var weightRow in weights)
        {
            writer.WriteLine(string.Join(",", weightRow));
        }
    }
}
// 8. load network weights and its positions 
static (int[] net, int[][] positions, float[][] weights) LoadPruningNetFromFile(string fileName)
{
    List<int[]> positionsList = new List<int[]>();
    List<float[]> weightsList = new List<float[]>();

    string[] lines = File.ReadAllLines(fileName);

    // Step 1: Read the network architecture (net array)
    int[] net = Array.ConvertAll(lines[0].Split(','), int.Parse);


    // Step 2: Read the positions array lines
    for (int i = 1; i < (lines.Length - 1) / 2 + 1; i++)
    {
        string line = lines[i];
        int[] positionRow = line == "" ? new int[0] : Array.ConvertAll(line.Split(','), int.Parse);
        positionsList.Add(positionRow);
        
    }
    for (int i = (lines.Length - 1) / 2 + 1; i < lines.Length; i++)
    {
        string line = lines[i];
        float[] weightRow = line == "" ? new float[0] : Array.ConvertAll(line.Split(','), float.Parse);
        weightsList.Add(weightRow);          
    }

    return (net, positionsList.ToArray(), weightsList.ToArray());
}
// todo: 9. fit the pruned architecture to used values only

// INFO
void PrintUserInfo()
{
#if DEBUG
Console.WriteLine("Debug mode is on, switch to Release mode");
#endif
    Console.WriteLine($"\nBegin neural network pruning demo\n");
}
void PrintHyperparameters()
{
    Console.WriteLine("Neural Network Configuration:");
    Console.WriteLine("NETWORK      = " + string.Join("-", net));
    Console.WriteLine("WEIGHTS      = " + net.Zip(net.Skip(1), (prev, curr) => prev * curr).Sum());
    Console.WriteLine("SEED         = " + SEED);
    Console.WriteLine("PRUNINGRATE  = " + PRUNING);
    Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    Console.WriteLine("EPOCHS       = " + EPOCHS);
    Console.WriteLine("FACTOR       = " + FACTOR);
    Console.WriteLine("PARALLEL     = " + PARALLEL);
}
static void PrintInfo(string str, int correct, int all, Stopwatch sw, bool showFPS = false)
{
    Console.WriteLine($"{str} Accuracy = {(correct * 100.0 / all).ToString("F2").PadLeft(6)}% | " +
       // $"Correct = {correct:N0}/{all:N0} | " + 
        $"Time = {(sw.Elapsed.TotalMilliseconds / 1000.0).ToString("F3")}s");

    if (showFPS)
        Console.WriteLine($"Test FPS = {10000 / sw.Elapsed.TotalSeconds:N0}");
}
// NETWORK AND DATA

struct AutoData
{
    public byte[] labelsTraining, labelsTest;
    public float[] samplesTrainingF, samplesTestF;

    static float[] NormalizeData(byte[] samples)
    {
        float[] samplesF = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            samplesF[i] = samples[i] / 255f;
        return samplesF;
    }

    public AutoData(string yourPath)
    {
        // Hardcoded URLs from my GitHub
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        byte[] test, training;

        // Change variable names for readability
        string trainDataPath = "trainData", trainLabelPath = "trainLabel", testDataPath = "testData", testLabelPath = "testLabel";

        if (!File.Exists(Path.Combine(yourPath, trainDataPath))
            || !File.Exists(Path.Combine(yourPath, trainLabelPath))
            || !File.Exists(Path.Combine(yourPath, testDataPath))
            || !File.Exists(Path.Combine(yourPath, testLabelPath)))
        {
            Console.WriteLine("Status: MNIST Dataset Not found");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // Padding bits: data = 16, labels = 8
            Console.WriteLine("Action: Downloading and Cleaning the Dataset from GitHub");
            training = new HttpClient().GetAsync(trainDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(60000 * 784).ToArray();
            labelsTraining = new HttpClient().GetAsync(trainLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(60000).ToArray();
            test = new HttpClient().GetAsync(testDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(10000 * 784).ToArray();
            labelsTest = new HttpClient().GetAsync(testLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(10000).ToArray();

            Console.WriteLine("Save Path: " + yourPath + "\n");
            File.WriteAllBytesAsync(Path.Combine(yourPath, trainDataPath), training);
            File.WriteAllBytesAsync(Path.Combine(yourPath, trainLabelPath), labelsTraining);
            File.WriteAllBytesAsync(Path.Combine(yourPath, testDataPath), test);
            File.WriteAllBytesAsync(Path.Combine(yourPath, testLabelPath), labelsTest);
        }
        else
        {
            // Data exists on the system, just load from yourPath
            Console.WriteLine("Dataset: MNIST (" + yourPath + ")" + "\n");
            training = File.ReadAllBytes(Path.Combine(yourPath, trainDataPath)).Take(60000 * 784).ToArray();
            labelsTraining = File.ReadAllBytes(Path.Combine(yourPath, trainLabelPath)).Take(60000).ToArray();
            test = File.ReadAllBytes(Path.Combine(yourPath, testDataPath)).Take(10000 * 784).ToArray();
            labelsTest = File.ReadAllBytes(Path.Combine(yourPath, testLabelPath)).Take(10000).ToArray();
        }

        samplesTrainingF = NormalizeData(training);
        samplesTestF = NormalizeData(test);
    }
}
