// Compile an .mlpackage to .mlmodelc next to it. Stands in for `xcrun coremlcompiler`,
// which ships with Xcode and not with the Command Line Tools.
import CoreML
import Foundation

let src = URL(fileURLWithPath: CommandLine.arguments[1])
let dst = src.deletingPathExtension().appendingPathExtension("mlmodelc")
let tmp = try MLModel.compileModel(at: src)
try? FileManager.default.removeItem(at: dst)
try FileManager.default.moveItem(at: tmp, to: dst)
print("compiled \(dst.path)")
