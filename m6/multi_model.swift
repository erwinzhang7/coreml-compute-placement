// In-process concurrency: K MLModel objects in ONE process, each driven by its own thread
// in a closed loop. Swift rather than Python so the GIL cannot serialise the callers and
// make two engines look like one.
//
//   multi_model <model.mlmodelc> <k> <seconds> [ne|gpu|all] [--shared]
//
// --shared loads one MLModel and drives it from K threads instead of K models.
// Prints one JSON line: per-thread calls and img/s, plus the total.
import CoreML
import Foundation

let a = CommandLine.arguments
guard a.count >= 4 else {
    FileHandle.standardError.write("usage: multi_model <mlmodelc> <k> <seconds> [ne|gpu|all] [--shared]\n".data(using: .utf8)!)
    exit(2)
}
let url = URL(fileURLWithPath: a[1])
let k = Int(a[2])!
let seconds = Double(a[3])!
let unitArg = a.count > 4 ? a[4] : "ne"
let shared = a.contains("--shared")

let cfg = MLModelConfiguration()
switch unitArg {
case "gpu": cfg.computeUnits = .cpuAndGPU
case "all": cfg.computeUnits = .all
default:    cfg.computeUnits = .cpuAndNeuralEngine
}

let nModels = shared ? 1 : k
let tLoad = Date()
let models = (0..<nModels).map { _ in try! MLModel(contentsOf: url, configuration: cfg) }
let loadS = Date().timeIntervalSince(tLoad)

let inDesc = models[0].modelDescription.inputDescriptionsByName.values.first!
let c = inDesc.multiArrayConstraint!
let shape = c.shape
let batch = shape[0].intValue

func makeInput() -> MLFeatureProvider {
    let arr = try! MLMultiArray(shape: shape, dataType: c.dataType)
    let n = arr.count
    if c.dataType == .float16 {
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        for i in 0..<n { p[i] = Float16(Float(i % 251) / 251.0).bitPattern }
    } else {
        for i in 0..<n { arr[i] = NSNumber(value: Float(i % 251) / 251.0) }
    }
    return try! MLDictionaryFeatureProvider(dictionary: [inDesc.name: MLFeatureValue(multiArray: arr)])
}
let inputs = (0..<k).map { _ in makeInput() }

// Warm every model (first predictions include ANE program load).
for m in models { for _ in 0..<5 { _ = try! m.prediction(from: inputs[0]) } }

var calls = [Int](repeating: 0, count: k)
var spans = [Double](repeating: 0, count: k)
let group = DispatchGroup()
let start = DispatchSemaphore(value: 0)
let lock = NSLock()
for t in 0..<k {
    group.enter()
    Thread {
        let m = models[shared ? 0 : t]
        start.wait()
        let t0 = Date()
        var n = 0
        while Date().timeIntervalSince(t0) < seconds {
            _ = try! m.prediction(from: inputs[t])
            n += 1
        }
        let dt = Date().timeIntervalSince(t0)
        lock.lock(); calls[t] = n; spans[t] = dt; lock.unlock()
        group.leave()
    }.start()
}
for _ in 0..<k { start.signal() }
group.wait()

let per = (0..<k).map { Double(calls[$0] * batch) / spans[$0] }
let total = per.reduce(0, +)
let perStr = per.map { String(format: "%.2f", $0) }.joined(separator: ",")
print("{\"k\":\(k),\"shared\":\(shared),\"units\":\"\(unitArg)\",\"batch\":\(batch),\"seconds\":\(seconds),\"load_s\":\(String(format: "%.2f", loadS)),\"calls\":\(calls),\"per_img_s\":[\(perStr)],\"total_img_s\":\(String(format: "%.2f", total))}")
