// Lists the compute devices Core ML reports. On a dual-ANE chip the question is whether
// this shows one Neural Engine with 32 cores or two with 16 each.
import CoreML
import Foundation

for d in MLComputeDevice.allComputeDevices {
    switch d {
    case .neuralEngine(let ne): print("neuralEngine totalCoreCount=\(ne.totalCoreCount)")
    case .gpu(let g):           print("gpu \(g.metalDevice.name)")
    case .cpu:                  print("cpu")
    @unknown default:           print("unknown \(d)")
    }
}
