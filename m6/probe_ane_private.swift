// Read-only look at what the private AppleNeuralEngine framework says about the engines.
// It calls only class-level getters that return device facts, and lists method names on
// the client classes to see whether any of them take an engine/device index. It loads
// nothing onto the ANE and changes no state. No entitlement, SIP or sudo needed.
import Foundation
import ObjectiveC

let fw = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
guard dlopen(fw, RTLD_NOW) != nil else { print("dlopen failed: \(String(cString: dlerror()))"); exit(1) }

func classGetter(_ cls: AnyClass, _ sel: String) -> String {
    let s = NSSelectorFromString(sel)
    guard let m = class_getClassMethod(cls, s) else { return "-" }
    let enc = String(cString: method_getTypeEncoding(m)!)
    let ret = enc.first!
    let imp = method_getImplementation(m)
    switch ret {
    case "@":
        typealias F = @convention(c) (AnyClass, Selector) -> AnyObject?
        return String(describing: unsafeBitCast(imp, to: F.self)(cls, s) ?? "nil" as NSString)
    case "B", "c":
        typealias F = @convention(c) (AnyClass, Selector) -> Bool
        return String(unsafeBitCast(imp, to: F.self)(cls, s))
    case "q", "Q", "i", "I", "l", "L":
        typealias F = @convention(c) (AnyClass, Selector) -> Int
        return String(unsafeBitCast(imp, to: F.self)(cls, s))
    default:
        return "(type \(enc), not called)"
    }
}

func methods(_ cls: AnyClass, meta: Bool) -> [String] {
    var n: UInt32 = 0
    let target: AnyClass = meta ? object_getClass(cls)! : cls
    guard let list = class_copyMethodList(target, &n) else { return [] }
    defer { free(list) }
    return (0..<Int(n)).map { NSStringFromSelector(method_getName(list[$0])) }.sorted()
}

if let info = NSClassFromString("_ANEDeviceInfo") {
    print("== _ANEDeviceInfo class getters")
    for m in methods(info, meta: true) where !m.contains(":") {
        print("  +\(m) = \(classGetter(info, m))")
    }
} else {
    print("_ANEDeviceInfo not found")
}

let interesting = ["engine", "device", "core", "cluster", "index", "balanc", "instance", "affinity", "priority", "queue"]
for name in ["_ANEClient", "_ANEModel", "_ANERequest", "_ANEDaemonConnection", "_ANEInMemoryModel",
             "_ANEProgramForEvaluation", "_ANEQoSMapper", "_ANEDeviceController", "_ANEVirtualClient"] {
    guard let c = NSClassFromString(name) else { print("== \(name): not present"); continue }
    let all = methods(c, meta: false).map { "-" + $0 } + methods(c, meta: true).map { "+" + $0 }
    let hits = all.filter { m in interesting.contains { m.lowercased().contains($0) } }
    print("== \(name): \(all.count) methods; matching \(interesting):")
    for h in hits { print("  \(h)") }
}
