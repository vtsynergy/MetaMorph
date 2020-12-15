/** \file
 *
 * \brief An OpenCL host-code boilerplate and interface generator
 * \copyright 2018-2021 Virginia Tech
 *
 *   This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 *
 *   This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

 *   You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
 *
 * A copy of the current license terms may be obtained from https://github.com/vtsynergy/MetaMorph/blob/master/LICENSE
 *
 *
 * MetaCL
 * A tool to consume OpenCL kernel files and produce MetaMorph-compatible
 * host-side wrappers for the contained kernels.
 *
 * BETA/Prototype software, no warranty expressed or implied.
 *
 * \author Paul Sathre
 * \author Atharva Gondhalekar
 *
 */

#include "clang/AST/Attr.h" //This file has the auto-generated implementations of all the attribute classes, reference it if you need to get information out of an attribute
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

/** Linked files to export directly */
extern "C" const char _binary_metamorph_emulatable_h_start[];
extern "C" const char _binary_metamorph_emulatable_h_end[];
extern "C" const char _binary_metamorph_opencl_emulatable_h_start[];
extern "C" const char _binary_metamorph_opencl_emulatable_h_end[];
extern "C" const char _binary_metamorph_shim_c_start[];
extern "C" const char _binary_metamorph_shim_c_end[];
extern "C" const char _binary_shim_dynamic_h_start[];
extern "C" const char _binary_shim_dynamic_h_end[];

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

/** Generate a simple OpenCL error check and linedump string for inclusion in
 * output code IFF InlineErrorCheck=true, else an empty string */
#define ERROR_CHECK(indent, errorcode, text)                                   \
  (InlineErrorCheck.getValue() ? indent                                        \
       "if (" errorcode " != CL_SUCCESS) fprintf(stderr, \"" text              \
       " \%d at \%s:\%d\\n\", " errorcode ", __FILE__, __LINE__);\n"           \
                               : "")
/** \deprecated a shorthand way to append a ternary-checked generated error-check string to an existing std::string buffer */
#define APPEND_ERROR_CHECK(string, indent, errorcode, text)                    \
  string += ERROR_CHECK(indent, errorcode, text)

static llvm::cl::OptionCategory MetaCLCategory("MetaCL Options");

/** A command line option to specify that all input .cl files should be wrapped
 * by a single pair of host .h/.c files, and the name to use. Defaults to
 * non-unified
 * \return An Option type that can be queried for the name of the unified output
 * files or "" if none is specified
 */
llvm::cl::opt<std::string, false> UnifiedOutputFile(
    "unified-output-file", /// < The option's name
    llvm::cl::desc(
        "If a filename is provided, all kernel files will generate a single "
        "set of host wrappers, instead of one per file."), /// < The option's
                                                           /// help text
    llvm::cl::value_desc("\"filename\""), /// < The option's value description
    llvm::cl::init(""), /// < The option defaults to an empty string (which is
                        /// treated as non-unified mode)
    llvm::cl::cat(MetaCLCategory));
/**
 * A command line option to additionally generate separate argument assignment
 * and enqueue functions. Useful if no arguments need to be reset in iterative
 * invocations or if subsets of arguments are assigned at different times.
 * Defaults to false
 * \return An Option type that can be queried for whether split wrappers should
 * additionally be generated
 */
llvm::cl::opt<bool, false> SplitWrappers(
    "split-wrappers", /// < The name of the option
    llvm::cl::desc(
        "In addition to combined args+launch wrapper, generate separate "
        "argument assignment and enqueue wrappers."), /// The option's help text
    llvm::cl::value_desc(
        "true/false"),     /// < Description of the option as a boolean
    llvm::cl::init(false), /// < The option defaults to true
    llvm::cl::cat(MetaCLCategory));
/**
 * A command line option to control whether enqueue sizes are in OpenCL
 * global/local worksize model (default) or CUDA grid/block model
 * \return An Option type that can be queried for whether split wrappers should
 * additionally be generated
 */
llvm::cl::opt<bool, false> GridBlockSizes(
    "cuda-grid-block", /// < The name of the option
    llvm::cl::desc(
        "Use CUDA Grid/Block model for kernel launch parameters"), /// The
                                                                   /// option's
                                                                   /// help text
    llvm::cl::value_desc(
        "true/false"),     /// < Description of the option as a boolean
    llvm::cl::init(false), /// < The option defaults to true
    llvm::cl::cat(MetaCLCategory));
/**
 * A command line option to control generation of error checks on all OpenCL
 * Runtime calls. Disabling may slightly reduce runtime overhead for
 * already-validated codes. Not recommended for development codes. Defaults to
 * true
 * \return An Option type that can be queried for whether inline error checking
 * should be performed
 */
llvm::cl::opt<bool, false> InlineErrorCheck(
    "inline-error-check", /// < The name of the option
    llvm::cl::desc("Generate an immediate error check after every OpenCL "
                   "runtime call."), /// The option's help text
    llvm::cl::value_desc(
        "true/false"),    /// < Description of the option as a boolean
    llvm::cl::init(true), /// < The option defaults to true
    llvm::cl::cat(MetaCLCategory));
/** A command line option to instruct MetaCL it is safe to overwrite existing
 * files with the same name as those that would be generated. If False, will
 * attempt to only replace generated code in same-named outputs.
 * \return An Option that can be queried for whether MetaCL should just
 * overwrite existing files with the same name as its output(s), or should
 * instead try to just replace generated code within identically-named files
 * \todo TODO Implement: For now effectively hardcoded to TRUE.
 */
llvm::cl::opt<bool, false> OverwriteFiles(
    "overwrite-files", /// < The option's name
    llvm::cl::desc("WIP NOOP: Instead of trying to replace only "
                   "MetaCL-generated code in existing output files, simply "
                   "overwrite them in entirety."), /// < The option's help text
    llvm::cl::value_desc("true/false"), /// < The option's value description
    llvm::cl::init(false), /// < Once implemeted the option will default to
                           /// false (try to replace a generated region of a
                           /// file, rather than the whole file)
    llvm::cl::cat(MetaCLCategory));
/** The three support levels for MetaMorph, the historical default would be
"required"
 * Explicitly assigned to support binary logic */
enum MetaMorphLevel {
  metaMorphDisabled = 2,
  metaMorphOptional = 3,
  metaMorphRequired = 1
};
/** A command line option to instruct MetaCL on whether the output code should
REQUIREi MetaMorph and the OpenCL backend, support it as an OPTIONAL plugin but
provide a non-MetaMorph fallback, or DISABLE MetaMorph aentirely and only use
the fallback.
 * The historical behavior is equivalent to REQUIRED, but OPTIONAL will be the
default going forward
 * \return An Option that can be queried for the enum value corresponding to the level of MetaMorph support needed
*/
llvm::cl::opt<MetaMorphLevel> UseMetaMorph(
    "use-metamorph", /// < The option's name
    llvm::cl::desc(
        "The level of MetaMorph integration in the output application."),
    llvm::cl::values(
        clEnumValN(
            metaMorphDisabled, "DISABLED",
            "Do not interoperate with MetaMorph at all, generate a shim that "
            "emulates the necessary functionality instead."),
        clEnumValN(
            metaMorphOptional, "OPTIONAL",
            "Default: Generate a shim file that can emulate the necessary "
            "MetaMorph functionality, but will attempt to dynamically-load "
            "MetaMorph and use it first, if available."),
        clEnumValN(metaMorphRequired, "REQUIRED",
                   "Former Default: Require that MetaMorph is present at "
                   "runtime and use it for all necessary functionality.")),
    llvm::cl::init(metaMorphOptional), llvm::cl::cat(MetaCLCategory));

/** The unified-output .c file's buffer */
raw_ostream *unified_output_c = NULL;
/** The unified-output .h file's buffer */
raw_ostream *unified_output_h = NULL;

/**
 * \brief Per-input-file storage struct for generated code
 *
 * Separately stores all the one-time init/deinit code for a given input
 *  in addition to vectors of all the cl_kernels and their respective init
 *  and deinit code.
 * Additionally, stores all generated wrapper functions and their respective
 *  prototypes, and pointers to the appropriate .h and.c output files, to be
 *  used when not running in unified output mode.
 */
// Meant to store all the boilerplate from a single input kernel file
typedef struct {
  /** metaOpenCLLoadProgramSource, clCreateProgram, clBuildProgram, cl_programs,
   * and associated internal stack management code */
  std::string runOnceInit;
  /** clReleaseProgram */
  std::string runOnceDeinit;
  /** All the global cl_kernel declarations for an input file */
  std::vector<std::string> cl_kernels;
  /** clCreateKernel and associated safety checks */
  std::vector<std::string> kernelInit;
  /** clReleaseKernel and associated checks*/
  std::vector<std::string> kernelDeinit;
  /** prototypes of host-side kernel wrappers, indexed by name, with arg string
   * as the value */
  std::vector<std::string> hostProtos;
  /** Implementations of host-side kernel wrappers (w/o prototypes), indexed by
   * host-side name, with function body as value */
  std::vector<std::string> hostFuncImpls;
  /** Output header file if not running in unified output mode */
  raw_ostream *outfile_h;
  /** Output implementation file if not running in unified output mode */
  raw_ostream *outfile_c;
} hostCodeCache;

/** Simple map of all hostCodeCaches (indexed by input filename) */
std::map<std::string, hostCodeCache *> AllHostCaches;

/**
 * Implementation of the MatchCallback class to handle all the code generation
 *  and storage necessary for a given kernel definition
 */
class PrototypeHandler : public MatchFinder::MatchCallback {
public:
  /**
   * Simple initialization constructor
   * \param c_outfile The output implementation buffer
   * \param h_outfile The output header buffer
   * \param infile The name of the file this kernel prototype was found in (with
   * path and extension removed)
   * \return a new prototype handler with output_file_c, output)file_h, and
   * filename appropriately initialized
   */
  PrototypeHandler(raw_ostream *c_outfile, raw_ostream *h_outfile,
                   std::string infile)
      : output_file_c(c_outfile), output_file_h(h_outfile), filename(infile) {
    // Any initialization?
  }
  virtual void run(const MatchFinder::MatchResult &Result);

  /**
   * \brief Simple struct to organize important details about a kernel parameter
   *
   */
  typedef struct {
    /** The type of a param as represented in the device AST */
    QualType devType;
    /** The host-compatible type mapped from devType */
    std::string hostType;
    /** The index of this kernel param */
    unsigned int devPos;
    /** \todo TODO add other flags (restrict, addrspace, etc) */
    /** if it is local memory we need the size, not a pointer */
    bool isLocal = false;
    /** if it is global or const, we need a pointer (cl_mem) */
    bool isGlobalOrConst = false;
    /** The param name, for mapping to the host wrapper prototype */
    std::string name;
  } argInfo;

private:
  std::string filename = "";
  raw_ostream *output_file_c = NULL;
  raw_ostream *output_file_h = NULL;
};

namespace std {
/**
 * Enforce a simple ordering on QualTypes so they can be used as std::map keys
 * \todo FIXME this sort was arbitrarily chosen, replace it with something else
 * if it allows types to be used before defined
 */
template <> struct less<QualType> {
  /** Return true if the OpaquePtr for first is less than second's
   * \param first The first QualType to compare
   * \param second The second QualType to compare
   * \return Whether first's opaque pointer is less than second's
   */
  bool operator()(const QualType first, const QualType second) {
    return (first.getAsOpaquePtr() < second.getAsOpaquePtr());
  }
};
} // namespace std

/** Internal storage of generated host-side record and typedef declarations and
 * definitions The outer map is indexed by output header file, and the inner map
 * is indexed by the device-side QualType
 */
std::map<raw_ostream *, std::map<QualType, std::string>> ImportedTypes;
/** Get the host-side name for the device-side QualType
 * \param type The Device QualType to convert to a host-compatible type string
 * \return the host-side name of the imported type
 */
std::string getHostType(QualType type);

/** For a given Qualtype will do a postorder traversal of all required records
 * and typedefs to ensure they are generated in the given output header file
 * \param type The Device Qualtype to inspect and import (if necessary)
 * \param Context The Device ASTContext from which type was pulled
 * \param out_header The output header's buffer (used to index into
 * ImportedTypes to ensure each user-defined type is only imported once per
 * output header)
 * \param alreadyImported When recursively importing types with dependencies,
 * inform the callee if the caller is already responsible for importing a type
 * with the same name as the callee (for example in certain typedefd structs the
 * typedef and struct have the same name and should only be imported once as a
 * combined code element)
 * \return The host-side equivalent type as a string
 * \post ImportedTypes contains (if it does not already) map of imported types
 * for the provided output header, which itself contains the host-side
 * representation of the provided device QualType and all its dependencies)
 */
std::string importTypeDependencies(QualType type, ASTContext &Context,
                                   raw_ostream *out_header,
                                   bool alreadyImported = false) {
  // If it's a primitive type, don't store anything but convert it to the OpenCL
  // type
  if (isa<BuiltinType>(type)) {
    return getHostType(type);
  }

  // If it's an OpenCL "primitive" type with embedded typedef (like uchar -->
  // unsigned char) don't import it We can recognize these by their presence in
  // Clang's opencl-c.h file
  if (const TypedefType *t = dyn_cast<TypedefType>(type)) {
    if (Context.getSourceManager()
            .getBufferName(t->getDecl()->getLocation())
            .str()
            .find("opencl-c.h") != std::string::npos) {
      return getHostType(type);
    }
  }

  // If it's an array type, make sure we resolve the element types
  // dependencies/rewrites
  if (const ArrayType *a = dyn_cast<ArrayType>(type)) {
    std::string imported =
        importTypeDependencies(a->getElementType(), Context, out_header);
    // to ensure the type is rewritten properly we must reconstruct the array
    // type after import
    std::string type_str = type.getAsString();
    std::string elem_str = a->getElementType().getAsString();
    // if the imported type is an array, we don't need the extra size tags for
    // modifying our own return type
    std::string::size_type pos;
    if ((pos = elem_str.find(' ')) != std::string::npos)
      elem_str.erase(pos);
    if ((pos = imported.find(' ')) != std::string::npos)
      imported.erase(pos);
    type_str.replace(type_str.find(elem_str), elem_str.length(), imported);
    return type_str;
  }

  // If the specific type we need is not constructed yet
  if (ImportedTypes[out_header].find(type) == ImportedTypes[out_header].end()) {
    // We have to record that we've imported it before we recurse so that it
    // acts as it's own base case (else it'll recurse infinitely)
    (ImportedTypes[out_header])[type] = "";
    // Remember if an outer typdef is responsible for defining an interior
    // RecordType
    bool firstEncountered = false;
    // If this is a combined Typedef/Record decl, the internal recursions for
    // the ElaboratedType and RecordType do not need to re-import the text;
    if (isa<TypedefType>(type) &&
        isa<ElaboratedType>(type.getSingleStepDesugaredType(Context))) {
      // If this is the first time the internal elaborated struct is
      // encountered, remember that we need to implement it ourselves because it
      // will be inhibited
      firstEncountered =
          (ImportedTypes[out_header].find(type.getSingleStepDesugaredType(
               Context)) == ImportedTypes[out_header].end());
      importTypeDependencies(type.getSingleStepDesugaredType(Context), Context,
                             out_header, true);
    } else if (isa<ElaboratedType>(type) &&
               isa<RecordType>(type.getSingleStepDesugaredType(Context))) {
      importTypeDependencies(type.getSingleStepDesugaredType(Context), Context,
                             out_header, true);
    } else {
      // Remember the outer recursive instance's import status
      importTypeDependencies(type.getSingleStepDesugaredType(Context), Context,
                             out_header, alreadyImported);
    }
    if (const TypedefType *t = dyn_cast<TypedefType>(type)) {
      if (!alreadyImported) {
        // This elaborate if simply sorts out records that would be anonymous if
        // not for a surrounding typdef, which must be handled differently
        /// \todo TODO: support unions and enums, any types of records other than struct (Found in TagDecl)
        if (isa<ElaboratedType>(t->getDecl()->getUnderlyingType()) &&
            isa<RecordType>(type.getSingleStepDesugaredType(Context)
                                .getSingleStepDesugaredType(Context)) &&
            dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context)
                                     .getSingleStepDesugaredType(Context))
                    ->getDecl()
                    ->getTypedefNameForAnonDecl() != NULL) {
          std::string structName =
              dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context)
                                       .getSingleStepDesugaredType(Context))
                  ->getDecl()
                  ->getTypedefNameForAnonDecl()
#if (LLVM_VERSION_MAJOR < 11)
                  ->getName();
#else
                  ->getName().str();
#endif
          // This first branch deals with a singular typedef that defines
          // multiple types pointing to the same anonymous struct, all but the
          // first will be generated as separate decls
          if (structName != t->getDecl()->getName()) {
            std::string record = "typedef " + structName + " " +
                                 t->getDecl()->getName().str() + ";\n";
            (ImportedTypes[out_header])[type] = record;
          } else {
            // This branch handles the primary (which inherits its name from the
            // first typedef'd type construct the record in-place
            std::string record = "typedef struct {\n";
            for (const FieldDecl *field :
                 dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context)
                                          .getSingleStepDesugaredType(Context))
                     ->getDecl()
                     ->fields()) {
              if (isa<ArrayType>(field->getType())) {
                std::string arr_type = importTypeDependencies(
                    field->getType(), Context, out_header);

                record += "  " +
                          arr_type.insert(arr_type.find("["),
                                          field->getName().str()) +
                          ";\n";
              } else {
                record += "  " +
                          importTypeDependencies(field->getType(), Context,
                                                 out_header) +
                          " " + field->getName().str() + ";\n";
              }
            }
            record += "} " + type.getAsString() + ";\n";
            (ImportedTypes[out_header])[type] = record;
          }
        } else {
          // Typedefs of named types (either scalars or records)
          // If the underlying type is a record and hasn't been constructed yet,
          // it will be inhibited from generating itself so we have to handle it
          // here
          if (isa<ElaboratedType>(t->getDecl()->getUnderlyingType()) &&
              firstEncountered) {
            std::string record =
                "typedef " + t->getDecl()->getUnderlyingType().getAsString() +
                " {\n";
            for (const FieldDecl *field :
                 dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context)
                                          .getSingleStepDesugaredType(Context))
                     ->getDecl()
                     ->fields()) {
              if (isa<ArrayType>(field->getType())) {
                std::string arr_type = importTypeDependencies(
                    field->getType(), Context, out_header);

                record += "  " +
                          arr_type.insert(arr_type.find("["),
                                          field->getName().str()) +
                          ";\n";
              } else {
                record += "  " +
                          importTypeDependencies(field->getType(), Context,
                                                 out_header) +
                          " " + field->getName().str() + ";\n";
              }
            }
            (ImportedTypes[out_header])[type] =
                record + "} " + type.getAsString() + ";\n";

          } else {
            (ImportedTypes[out_header])[type] =
                "typedef " +
                importTypeDependencies(t->getDecl()->getUnderlyingType(),
                                       Context, out_header) +
                " " + type.getAsString() + ";\n";
          }
        }
      }
    }
    // If it's an ElaboratedType without a wrapping typedef
    if (const ElaboratedType *e = dyn_cast<ElaboratedType>(type)) {
      if (!alreadyImported) {
        // construct the record in-place
        std::string record = type.getAsString() + " {\n";
        for (const FieldDecl *field :
             dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context))
                 ->getDecl()
                 ->fields()) {
          record +=
              "  " +
              importTypeDependencies(field->getType(), Context, out_header) +
              " " + field->getName().str() + ";\n";
        }
        record += "};\n";
        (ImportedTypes[out_header])[type] = record;
      }
    }
    // If it's a Record without an explicit keyword (like struct or union)
    if (const RecordType *r = dyn_cast<RecordType>(type)) {
      if (!alreadyImported) {
        std::string record = type.getAsString() + " {\n";
        for (const FieldDecl *field : r->getDecl()->fields()) {
          record +=
              "  " +
              importTypeDependencies(field->getType(), Context, out_header) +
              " " + field->getName().str() + ";\n";
        }
        record += "};\n";
        (ImportedTypes[out_header])[type] = record;
      }
    }
  }
  // Anything that isn't caught by the builtin, OpenCL, and array filters is
  // going to preserve it's own name
  return type.getAsString();
}
std::string getHostType(QualType devType) {
  std::string retType = "";
  /// \todo TODO handle types that don't directly map to a cl_type (images, vectors)
  retType += "cl_";
  std::string canon = devType.getAsString();
  std::string::size_type pos;
  if ((pos = canon.find("unsigned ")) != std::string::npos)
    canon.replace(pos, 9, "u");
  // strip off and appropriately modify vectorization attributes
  // "__attribute__((ext_vector_type(<width>))) Assumes the intervening pointer
  // type has already been removed
  if ((pos = canon.find(" __attribute__((ext_vector_type(")) !=
      std::string::npos) {
    std::string::size_type endPos = canon.find(")))", pos);
    canon.erase(endPos, 3);
    canon.erase(pos, 32);
  }

  /// \todo FIXME Technically Clang should catch bool struct elements (and it does if it's directly a parameter, but not if the param is a pointer to a struct with a bool in it)
  if (canon == "_Bool")
    return "\n#error passing a boolean through a pointer or struct pointer in "
           "OpenCL is undefined\ncl_" +
           canon;

  return "cl_" + canon;
}

/**
 * Fully analyze a device kernel parameter and cache any necessary generated
 * host-side user-defined types
 * \param devParam A device kernel parameter to analyze
 * \param Context the ASTContext from which we pulled the devParam
 * \param stream the output header file's buffer for caching any necessary
 * dependent types
 * \post any user-defined type dependencies needed for the devParam are imported
 * to the output host header file (if not already)
 * \return a data structure describing the host-requirements of the parameter's
 * device type
 */
PrototypeHandler::argInfo *analyzeDevParam(ParmVarDecl *devParam,
                                           ASTContext &Context,
                                           raw_ostream *stream) {
  PrototypeHandler::argInfo *retArg = new PrototypeHandler::argInfo();
  // Detect the type of the device parameter
  retArg->devType = devParam->getType();
  // If it is a pointer type, add address space flags
  /// \todo FIXME Deal with NULL TypePtr
  if (retArg->devType->isAnyPointerType()) {
    clang::LangAS addrSpace =
        retArg->devType->getPointeeType().getAddressSpace();
    if (addrSpace == LangAS::opencl_local)
      retArg->isLocal = true;
    if (addrSpace == LangAS::opencl_global ||
        addrSpace == LangAS::opencl_constant)
      retArg->isGlobalOrConst = true;
    retArg->hostType = importTypeDependencies(
        retArg->devType->getPointeeType().getUnqualifiedType(), Context,
        stream);
    // retArg->hostType = getHostType(retArg->devType);
  } else {
    retArg->hostType = importTypeDependencies(
        retArg->devType.getUnqualifiedType(), Context, stream);
  }
  // borrow the name
  retArg->name = devParam->getNameAsString();
  /// \todo TODO grab any additional param qualifiers we need to keep in mind (restrict, constant, address space etc)
  return retArg;
}

/**
 * Remove the leading path and tailing type extension from a filename
 * \param in A full filename with type extension and path
 * \return Just the filename with type and path information removed
 * \todo TODO handle filenames without a forward slash or period
 */
std::string trimFileSlashesAndType(std::string in) {
  size_t dotPos = in.rfind('.');
  size_t slashPos = in.rfind('/') + 1;
  return in.substr(slashPos, dotPos - slashPos);
}

/**
 * \brief Generate host wrapper, init, and deinit for each found kernel
 * definition
 *
 * For each Matched kernel definition:
 *  Create a cl_kernel based on input filename and kernel name
 *  Create appropriate clCreateKernel initialization and safety checks
 *  Create appropriate clReleaseKernel deinit and safety checks
 *  Create a host side kernel wrapper with all clSetKernelArg and clEnqueue
 * calls and associated safety checks
 * \param Result An ASTMatch for a kernel function prototype to process
 * \post The hostCodeCache object in AllHostCaches[this::filename] has a
 * cl_kernel and associated wrapper, initialization, and deinitialization
 * boilerplate added to it
 */
void PrototypeHandler::run(const MatchFinder::MatchResult &Result) {
  bool singleWorkItem = false;
  const FunctionDecl *func;
  const FunctionDecl *nd_func = NULL;
  std::string outFile =
      ((UnifiedOutputFile.getValue() == "") ? filename
                                            : UnifiedOutputFile.getValue());
  if (func = Result.Nodes.getNodeAs<FunctionDecl>("swi_kernel_def")) {
    singleWorkItem = true;
    nd_func = Result.Nodes.getNodeAs<FunctionDecl>("nd_func");
  } else
    func = Result.Nodes.getNodeAs<FunctionDecl>("kernel_def");
  if (func) {
    // Figure out the host-side name
    /// \todo TODO Proposed Feature: If the name of the kernel ends in "_<cl_type>", strip it and register it as an explicitly typed kernel (for sharing a host wrapper) for all other cases, create a new meta_typeless type
    std::string host_func =
        "metacl_" + filename + "_" + func->getNameAsString();
    /// \todo TODO hoist attribute handling out to it's own function that returns a single data structure with all the kernel attributes we might need to handle
    unsigned int work_group_size[4] = {
        1, 1, 1, 0}; // 4th member is the type of the size (0 = unspecified, 1 =
                     // hint, 2 = required, 3 = intel required)
    for (auto attr : func->getAttrs()) {
      /// \todo TODO Xilinx adds the xcl_max_work_group_size and xcl_zero_global_work_offset attributes to kernels
      /// \todo TODO clang doesn't appear to have the OpenCL pointer nosvm attribute yet
      /// \todo TODO recognize the endian attribute so that if it is explicitly specified, we can warn in the host API
      /// \todo TODO implement handlers for any necessary kernel type attributes
      if (VecTypeHintAttr *vecAttr = dyn_cast<VecTypeHintAttr>(attr)) {
        /// \todo TODO do something with VecTypeHintAttrs
        vecAttr->getTypeHint().getAsString();
      } else if (WorkGroupSizeHintAttr *sizeAttr =
                     dyn_cast<WorkGroupSizeHintAttr>(attr)) {
        if (work_group_size[3] <
            2) { // Only if we don't already have a required size
          work_group_size[0] = sizeAttr->getXDim(),
          work_group_size[1] = sizeAttr->getYDim(),
          work_group_size[2] = sizeAttr->getZDim(), work_group_size[3] = 1;
        }
        llvm::errs() << "Suggested work group size is (" << work_group_size[0]
                     << ", " << work_group_size[1] << ", " << work_group_size[2]
                     << ")\n";
      } else if (ReqdWorkGroupSizeAttr *sizeAttr =
                     dyn_cast<ReqdWorkGroupSizeAttr>(attr)) {
        if (work_group_size[3] <
            2) { // Only if we don't already have a required size
          work_group_size[0] = sizeAttr->getXDim(),
          work_group_size[1] = sizeAttr->getYDim(),
          work_group_size[2] = sizeAttr->getZDim(), work_group_size[3] = 2;
        }
        llvm::errs() << "Required work group size is (" << work_group_size[0]
                     << ", " << work_group_size[1] << ", " << work_group_size[2]
                     << ")\n";
      } else if (OpenCLIntelReqdSubGroupSizeAttr *subSize =
                     dyn_cast<OpenCLIntelReqdSubGroupSizeAttr>(attr)) {
        /// \todo TODO Handle OpenCLIntelReqdWubGroupSizeAttr
      } /// \todo TODO Handle other important kernel attributes
    }
    /// \todo TODO implement asynchronous call API (allow wrappers to wait on kernels as well as return event type)
    /// \todo TODO check the prototype for any remaining OpenCL-specific attributes that require the host to behave in a particular way

    // Creating the AST consumer forces all the once-per-input boilerplate to be
    // generated so we don't have to do it here
    hostCodeCache *cache = AllHostCaches[filename];
    std::string framed_kernel =
        "frame->" + filename + "_" + func->getNameAsString() + "_kernel";
    std::string current_kernel =
        "__metacl_" + outFile + "_current_" + framed_kernel;
    cache->cl_kernels.push_back("  cl_kernel " + filename + "_" +
                                func->getNameAsString() + "_kernel;\n");
    // Generate a clCreatKernelExpression
    cache->kernelInit.push_back(
        "    " + current_kernel + " = clCreateKernel(__metacl_" + outFile +
        "_current_frame->" + filename + "_prog, \"" + func->getNameAsString() +
        "\", &createError);\n");
    cache->kernelInit.push_back(
        ERROR_CHECK("    ", "createError", "OpenCL kernel creation error"));
    // Generate a clReleaseKernelExpression
    cache->kernelDeinit.push_back("      releaseError = clReleaseKernel(" +
                                  current_kernel + ");\n");
    cache->kernelDeinit.push_back(
        ERROR_CHECK("      ", "releaseError", "OpenCL kernel release error"));

    // Strings to assemble common elements
    // Doxygen header(s)
    std::string doxygenGeneral = "", doxygenArgs = "", doxygenEnqueue = "";
    // The launch parameter list
    std::string launchParams = "";
    // The argument assignment parameter list
    std::string kernargParams = "";
    // The registration/initialization check
    std::string regCheck = "";
    // The worksize computation/validation
    std::string sizeCheck = "";
    // The actual argument assignments
    std::string setArgs = "";
    // The actual kernel invocation and completion
    std::string enqueue = "";
    // The kernel wrapping function(s) are completely assembled later

    // Begin constructing the host wrapper

    // Construct general doxygen
    doxygenGeneral += "/** Automatically-generated by MetaCL\n";
    if (singleWorkItem)
      doxygenGeneral += "Kernel function is detected as Single-Work-Item\n";
    doxygenGeneral +=
        "\\param queue the cl_command_queue the kernel is being prepared to "
        "run on (to lookup and/or enqueue associated cl_kernel)\n";

    // Construct the registration and initialization check
    // Add module-registration check/lazy registration
    regCheck += "  if (metacl_" + outFile +
                "_registration == NULL) meta_register_module(&metacl_" +
                outFile + "_registry);\n";
    // Query the frame to launch the kernel on from the queue
    regCheck += "  struct __metacl_" + outFile + "_frame * frame = __metacl_" +
                outFile + "_current_frame;\n";
    regCheck += "  if (queue != NULL) frame = __metacl_" + outFile +
                "_lookup_frame(queue);\n";
    // NULL check that the frame is valid
    regCheck += "  //If the user requests a queue this module doesn't know "
                "about, or a NULL queue and there is no current frame\n";
    regCheck += "  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;\n";
    // Add a per-program check for initialization
    regCheck +=
        "  if (frame->" + filename + "_init != 1) return CL_INVALID_PROGRAM;\n";
    regCheck += "  cl_int retCode = CL_SUCCESS;\n";

    // Construct argument handling code
    int pos = 0;
    for (ParmVarDecl *parm : func->parameters()) {
      // Figure out the host-side representation of the param
      PrototypeHandler::argInfo *info =
          analyzeDevParam(parm, *Result.Context, output_file_h);
      // Add it to the wrapper param list
      // If it's not local, directly use the host type and get/set the data
      // itself
      if (info->isGlobalOrConst) {
        kernargParams += ", cl_mem * " + info->name;
        setArgs += "  retCode = clSetKernelArg(" + framed_kernel + ", " +
                   std::to_string(pos) + ", sizeof(cl_mem), " + info->name +
                   ");\n";
        setArgs += ERROR_CHECK(
            "  ", "retCode",
            "OpenCL kernel argument assignment error (arg: \\\"" + info->name +
                "\\\", host wrapper: \\\"" + host_func + "\\\")");
        doxygenArgs += "\\param " + info->name +
                       " a cl_mem buffer, must internally store " +
                       info->hostType + " types\n";
      } else if (info->isLocal) { // If it is local, instead create a size
                                  // variable and set the size of the memory
                                  // region
        kernargParams += ", size_t " + info->name + "_num_local_elems";
        setArgs += "  retCode = clSetKernelArg(" + framed_kernel + ", " +
                   std::to_string(pos) + ", sizeof(" + info->hostType + ") * " +
                   info->name + "_num_local_elems, NULL);\n";
        setArgs += ERROR_CHECK(
            "  ", "retCode",
            "OpenCL kernel argument assignment error (arg: \\\"" + info->name +
                "\\\", host wrapper: \\\"" + host_func + "\\\")");
        doxygenArgs +=
            "\\param " + info->name +
            "_num_local_elems allocate __local memory space for this many " +
            info->hostType + " elements\n";
      } else {
        kernargParams += ", " + info->hostType + " " + info->name;
        // generate a clSetKernelArg expression
        setArgs += "  retCode = clSetKernelArg(" + framed_kernel + ", " +
                   std::to_string(pos) + ", sizeof(" + info->hostType + "), &" +
                   info->name + ");\n";
        setArgs += ERROR_CHECK(
            "  ", "retCode",
            "OpenCL kernel argument assignment error (arg: \\\"" + info->name +
                "\\\", host wrapper: \\\"" + host_func + "\\\")");
        doxygenArgs += "\\param " + info->name +
                       " scalar parameter of type \"" + info->hostType + "\"\n";
      }
      pos++;
    }

    // At this point we have everything we need to finalize the setArg-only
    // wrapper, do so if we're running with SplitWrappers=true;
    if (SplitWrappers.getValue()) {
      // First, construct the prototype
      std::string setArgWrapper = "cl_int " + host_func +
                                  "_set_args(cl_command_queue queue" +
                                  kernargParams + ")";
      // Then add it to the header with the doxygen elements
      cache->hostProtos.push_back(doxygenGeneral + doxygenArgs + " */\n" +
                                  setArgWrapper + ";\n");
      // Then start assembling the implementation
      setArgWrapper += " {\n";
      // Need the registration and initialization checks in place
      setArgWrapper += regCheck;
      // Then we need to assign each of the arguments
      setArgWrapper += setArgs;
      // Then finalize the function
      setArgWrapper += "  return retCode;\n}\n";
      cache->hostFuncImpls.push_back(setArgWrapper + "\n");
    }

    // Construct the kernel enqueue code
    /// \todo FIXME Once we have a good way of dealing with explicitly-typed kernels re-add the meta_type_id parameter TODO add a meta_typeless type to metamorph's type enum
    //    doxygen += "\\param type the MetaMorph type of the function\n";
    // Add typical sizing parameters to the launch parameter list
    std::string outerSizeName, innerSizeName;
    if (GridBlockSizes.getValue()) {
      outerSizeName = "grid_size", innerSizeName = "block_size";
      //doxygenEnqueue += "\\param grid_size a size_t[3] providing the number of workgroups in the X and Y dimensions, and the number of iterations in the Z dimension\n";
      doxygenEnqueue += "\\param grid_size a size_t[3] providing the number of "
                        "workgroups in the X, Y, Z dimensions\n";
      doxygenEnqueue += "\\param block_size a size_t[3] providing the "
                        "workgroup size in the X, Y, Z dimensions";
    } else {
      outerSizeName = "global_size", innerSizeName = "local_size";
      doxygenEnqueue += "\\param global_size a size_t[3] providing the global "
                        "number of workitems in the X, Y, Z dimensions\n";
      doxygenEnqueue += "\\param local_size a size_t[3] providing the "
                        "workgroup size in the X, Y, Z dimensions";
    }
    if (singleWorkItem) {
      doxygenEnqueue += " (detected as single-work-item, must be {1, 1, 1})\n";
    } else if (work_group_size[3] == 1) {
      doxygenEnqueue += " (work_group_size_hint attribute suggests {" +
                        std::to_string(work_group_size[0]) + ", " +
                        std::to_string(work_group_size[1]) + ", " +
                        std::to_string(work_group_size[2]) + "})\n";
    } else if (work_group_size[3] == 2) {
      doxygenEnqueue += " (reqd_work_group_size attribute requires {" +
                        std::to_string(work_group_size[0]) + ", " +
                        std::to_string(work_group_size[1]) + ", " +
                        std::to_string(work_group_size[2]) + "})\n";
    } else {
      doxygenEnqueue += "\n";
    }
    launchParams +=
        ", size_t (*" + outerSizeName + ")[3], size_t (*" + innerSizeName +
        ")[3], size_t (*meta_offset)[3], int async, cl_event * event";
    doxygenEnqueue += "\\param meta_offset the NDRange offset, NULL if none\n";
    doxygenEnqueue +=
        "\\param async whether the kernel should run asynchronously\n";
    doxygenEnqueue += "\\param event returns the cl_event corresponding to the "
                      "kernel launch if run asynchronously\n";

    // Assemble the worksize checking code
    // Add pseudo auto-scaling safety code
    sizeCheck += "  meta_bool nullBlock = (" + innerSizeName +
                 " != NULL && (*" + innerSizeName + ")[0] == 0 && (*" +
                 innerSizeName + ")[1] == 0 && (*" + innerSizeName +
                 ")[2] == 0);\n";
    sizeCheck += "  size_t _global_size[3];\n";
    if (work_group_size[3] == 0 && !singleWorkItem) {
      sizeCheck +=
          "  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;\n";
    } else if (singleWorkItem) {
      sizeCheck += "  size_t _local_size[3] = {1, 1, 1};\n";
    } else {
      sizeCheck += "  size_t _local_size[3] = {" +
                   std::to_string(work_group_size[0]) + ", " +
                   std::to_string(work_group_size[1]) + ", " +
                   std::to_string(work_group_size[2]) + "};\n";
    }
    sizeCheck += "  size_t _temp_offset[3];\n";
    sizeCheck += "  if (meta_offset == NULL) { _temp_offset[0] = 0, "
                 "_temp_offset[1] = 0, _temp_offset[2] = 0;}\n";
    sizeCheck +=
        "  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = "
        "(*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }\n";
    sizeCheck += "  const size_t _meta_offset[3] = {_temp_offset[0], "
                 "_temp_offset[1], _temp_offset[2]};\n";
    sizeCheck += "  int iters;\n\n";
    sizeCheck += "  //Default runs a single workgroup\n";
    sizeCheck += "  if (" + outerSizeName + " == NULL || " + innerSizeName +
                 " == NULL) {\n";
    sizeCheck += "    _global_size[0] = _local_size[0];\n    _global_size[1] = "
                 "_local_size[1];\n    _global_size[2] = _local_size[2];\n";
    sizeCheck += "    iters = 1;\n";
    sizeCheck += "  } else {\n";
    if (work_group_size[3] == 1) {
      sizeCheck += "    if (!(_local_size[0] == (*" + innerSizeName +
                   ")[0] && _local_size[1] == (*" + innerSizeName +
                   ")[1] && _local_size[2] == (*" + innerSizeName +
                   ")[2])) {\n";
      sizeCheck += "      fprintf(stderr, \"Warning: kernel " +
                   func->getNameAsString() + " suggests a workgroup size of {" +
                   std::to_string(work_group_size[0]) + ", " +
                   std::to_string(work_group_size[1]) + ", " +
                   std::to_string(work_group_size[2]) +
                   "} at \%s:\%d\\n\", __FILE__, __LINE__);\n";
      sizeCheck += "    }\n";

    } else if (work_group_size[3] == 2 || singleWorkItem) {
      sizeCheck += "    if (!(_local_size[0] == (*" + innerSizeName +
                   ")[0] && _local_size[1] == (*" + innerSizeName +
                   ")[1] && _local_size[2] == (*" + innerSizeName + ")[2])";
      if (singleWorkItem)
        sizeCheck += " && !(_global_size[0] == 1 && _global_size[1] == 1 && "
                     "_global_size[2] == 1)";
      sizeCheck += ") {\n";
      sizeCheck += "      fprintf(stderr, \"Error: kernel " +
                   func->getNameAsString() + " requires a workgroup size of {" +
                   std::to_string(work_group_size[0]) + ", " +
                   std::to_string(work_group_size[1]) + ", " +
                   std::to_string(work_group_size[2]) +
                   "}, aborting launch at \%s:\%d\\n\", __FILE__, __LINE__);\n";
      sizeCheck += "      return CL_INVALID_WORK_GROUP_SIZE;\n";
      sizeCheck += "    }\n";
    }
    if (GridBlockSizes.getValue()) {
      sizeCheck += "    _global_size[0] = (*" + outerSizeName +
                   ")[0] * (nullBlock ? 1 : (*" + innerSizeName + ")[0]);\n";
      sizeCheck += "    _global_size[1] = (*" + outerSizeName +
                   ")[1] * (nullBlock ? 1 : (*" + innerSizeName + ")[1]);\n";
      sizeCheck += "    _global_size[2] = (*" + outerSizeName +
                   ")[2] * (nullBlock ? 1 : (*" + innerSizeName + ")[2]);\n";
    } else {
      // OpenCL will return CL_INVALID_WORK_GROUP_SIZE if it's not an even
      // divisor of global size, don't need to check
      sizeCheck += "    _global_size[0] = (*" + outerSizeName + ")[0];\n";
      sizeCheck += "    _global_size[1] = (*" + outerSizeName + ")[1];\n";
      sizeCheck += "    _global_size[2] = (*" + outerSizeName + ")[2];\n";
    }
    sizeCheck += "    _local_size[0] = (*" + innerSizeName + ")[0];\n";
    sizeCheck += "    _local_size[1] = (*" + innerSizeName + ")[1];\n";
    sizeCheck += "    _local_size[2] = (*" + innerSizeName + ")[2];\n";
    // sizeCheck += "    iters = (*_global_size_size)[2];\n";
    sizeCheck += "  }\n";

    /// \bug TODO workDim should not assume 3D kernels, we need to capture it from the kernel's attribute or the provided grid/block
    int workDim = 3;
    std::string globalSize = "_global_size",
                localSize = "(nullBlock ? NULL : _local_size)";
    /// \todo TODO expose and handle eventWaitLists and retEvents
    std::string eventWaitListSize = "0", eventWaitList = "NULL",
                retEvent = "event";
    if (singleWorkItem ||
        (work_group_size[3] != 0 && work_group_size[0] == 1 &&
         work_group_size[1] == 1 && work_group_size[2] == 1)) {
      enqueue += "  retCode = clEnqueueTask(frame->queue, " + framed_kernel +
                 ", " + eventWaitListSize + ", " + eventWaitList + ", " +
                 retEvent + ");\n";
      enqueue += ERROR_CHECK("  ", "retCode",
                             "OpenCL kernel enqueue error (host wrapper: \\\"" +
                                 host_func + "\\\")");
    } else {
      enqueue += "  retCode = clEnqueueNDRangeKernel(frame->queue, " +
                 framed_kernel + ", " + std::to_string(workDim) +
                 ", &_meta_offset[0], " + globalSize + ", " + localSize + ", " +
                 eventWaitListSize + ", " + eventWaitList + ", " + retEvent +
                 ");\n";
      enqueue += ERROR_CHECK("  ", "retCode",
                             "OpenCL kernel enqueue error (host wrapper: \\\"" +
                                 host_func + "\\\")");
    }
    enqueue += "  if (!async) {\n";
    enqueue += "    retCode = clFinish(frame->queue);\n";
    enqueue += ERROR_CHECK("    ", "retCode",
                           "OpenCL kernel execution error (host wrapper: \\\"" +
                               host_func + "\\\")");
    enqueue += "  }\n";

    // At this point we have everything we need to finalize the launch-only
    // wrapper, do so if we're running with SplitWrappers=true
    if (SplitWrappers.getValue()) {
      // First, construct the prototype
      std::string enqueueWrapper = "cl_int " + host_func +
                                   "_enqueue_again(cl_command_queue queue" +
                                   launchParams + ")";
      // Then add it to the header with the doxygen elements
      cache->hostProtos.push_back(doxygenGeneral + doxygenEnqueue + " */\n" +
                                  enqueueWrapper + ";\n");

      // Then start assembling the implementation
      enqueueWrapper += " {\n";
      // Need the registration and initialization checks in place
      enqueueWrapper += regCheck;
      // Then we need to check work sizes
      enqueueWrapper += sizeCheck;
      // Then launch the task/kernel and possibly wait for it to finish
      enqueueWrapper += enqueue;
      // Then finalize the function
      enqueueWrapper += "  return retCode;\n}\n";
      cache->hostFuncImpls.push_back(enqueueWrapper + "\n");
    }

    // We can now also construct the unified version
    std::string unifiedWrapper = "cl_int " + host_func +
                                 "(cl_command_queue queue" + launchParams +
                                 kernargParams + ")";
    // Then add it to the header with the doxygen elements
    cache->hostProtos.push_back(doxygenGeneral + doxygenEnqueue + doxygenArgs +
                                " */\n" + unifiedWrapper + ";\n");
    // Then assemble the implementation
    unifiedWrapper += " {\n";
    // Need the registration and initialization checks in place
    unifiedWrapper += regCheck;
    // Then we need to check work sizes
    unifiedWrapper += sizeCheck;
    // Then we need to assign each of the arguments
    unifiedWrapper += setArgs;
    // Then launch the task/kernel and possibly wait for it to finish
    unifiedWrapper += enqueue;
    // Then finalize the function
    unifiedWrapper += "  return retCode;\n}\n";
    cache->hostFuncImpls.push_back(unifiedWrapper + "\n");
  }
}

/** Simple override to consume kernel ASTs and look for kernels */
class KernelASTConsumer : public ASTConsumer {
public:
  /**
   * A simple ASTMatcher consumer to catch NDRange and Single Work Item Kernels
   * \param comp The compiler instance used to generate this AST
   * \param out_c This input file's output implementation buffer (could be
   * unified or per-input)
   * \param out_h This input file's output header buffer (could be unified or
   * per-input)
   * \param file The input filename, trimmed of leading path and trailing
   * filetype extension
   * \post Matcher has all our ASTMatchers added to it
   * \post CI contains a pointer to comp
   */
  KernelASTConsumer(CompilerInstance *comp, raw_ostream *out_c,
                    raw_ostream *out_h, std::string file)
      : CI(comp) {
    Matcher.addMatcher(
        /** Looking for functions that are either */
        functionDecl(anyOf(
            /** A single work item */
            functionDecl(
                allOf(hasAttr(attr::Kind::OpenCLKernel), isDefinition(),
                      /** If it calls these functions it won't be treated as SWI
                         by Intel/Altera */
                      unless(hasDescendant(callExpr(callee(
                          functionDecl(anyOf(hasName("get_global_id"),
                                             hasName("get_local_id"),
                                             hasName("get_local_linear_id"),
                                             hasName("get_group_id"),
                                             hasName("barrier")))
                              .bind("nd_func")))))))
                .bind("swi_kernel_def"),
            /** An NDRange */
            functionDecl(
                allOf(hasAttr(attr::Kind::OpenCLKernel), isDefinition()))
                .bind("kernel_def"))),
        /** When a kernel is found, pass it off to a handler */
        new PrototypeHandler(out_c, out_h, file));
  }
  /**
   * Simple override to trigger the ASTMatchers
   * \param Context the ASTContext we are handling
   * \post All kernel functions in the AST have been matched, callbacks
   * triggered, and appropriate host code generated and cached
   */
  void HandleTranslationUnit(ASTContext &Context) override {
    /** \todo TODO Any pre-match actions that need to take place? */
    Matcher.matchAST(Context);
  }

private:
  MatchFinder Matcher;
  CompilerInstance *CI;
};

/**
 * \brief Handler for a single .cl input file
 *
 * A custom class that allows us to do once-per-input things before
 *  handling individual kernels
 * Responsible for:
 *   creating output files in non-unified mode
 *   generating and storing init and deinit boilerplate in the global host
 *    code storage cache
 */
class MetaCLFrontendAction : public ASTFrontendAction {
public:
  MetaCLFrontendAction() {}

  //
  //    /** Currently unused mechanism for adding things before parsing */
  //    bool BeginInvocation(CompilerInstance &CompInst) {
  //      // \todo TODO, do we need to do anything before parsing like adding headers, etc?
  //      return ASTFrontendAction::BeginInvocation(CompInst);
  //    }
  //
  //    /** Currently unused mechanism for doing extra things after parsing */
  //    void EndSourceFileAction() {
  //      // \todo TODO what do we need to do at the end of a file
  //    }
  //
  /**
   * Create output files (if not unified) and all once-per-input boilerplate
   *  (i.e. context management structs, cl_programs, program init/deinit)
   * \param CI The CompilerInstance used to generate the AST we're going to
   * consumer
   * \param infile The input file that was parsed to generate the AST, with path
   * and filetype extension
   * \return a new KernelASTConsumer to Match all the kernel functions
   * \post AllHostCodeCaches[trimFileSlashesAndType(infile.str()) contains a
   * hostCodeCache with all the once-per-input-file boilerplate already
   * generated
   * \post If running in non-unified-output mode, buffers for both .c and .h
   * outputs with the same name as infile are created and added to the
   * hostCodeCache object
   */
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef infile) override {
    // generate the cl_program
    // At the beginning of processing each new file, create the associated
    // once-per-input-file boilerplate By looking up the host code cache in the
    // map, we force it to exist so we can populate it
    std::string file = trimFileSlashesAndType(infile.str());
    std::string outFile =
        ((UnifiedOutputFile.getValue() == "") ? file
                                              : UnifiedOutputFile.getValue());

    llvm::errs() << file << "\n";

    hostCodeCache *cache = AllHostCaches[file] = new hostCodeCache();
    // Add the core boilerplate to the hostCode cache
    /// \todo TODO add a function to metamorph for human-readable OpenCL error codes
    cache->runOnceInit +=
        "  if ((vendor & meta_cl_device_is_accel) && ((vendor & "
        "meta_cl_device_vendor_mask) == meta_cl_device_vendor_intelfpga)) {\n";
    /// \todo TODO enforce Intel name filtering to remove "kernel"
    cache->runOnceInit += "    __metacl_" + outFile + "_current_frame->" +
                          file + "_progLen = metaOpenCLLoadProgramSource(\"" +
                          file + ".aocx\", &__metacl_" + outFile +
                          "_current_frame->" + file + "_progSrc, &__metacl_" +
                          outFile + "_current_frame->" + file + "_progDir);\n";
    cache->runOnceInit += "    if (__metacl_" + outFile + "_current_frame->" +
                          file + "_progLen != -1)\n";
    cache->runOnceInit +=
        "      __metacl_" + outFile + "_current_frame->" + file +
        "_prog = clCreateProgramWithBinary(__metacl_" + outFile +
        "_current_frame->context, 1, &__metacl_" + outFile +
        "_current_frame->device, "
        "&__metacl_" +
        outFile + "_current_frame->" + file +
        "_progLen, (const unsigned char **)&__metacl_" + outFile +
        "_current_frame->" + file + "_progSrc, NULL, &buildError);\n";
    cache->runOnceInit += "  } else {\n";
    cache->runOnceInit += "    __metacl_" + outFile + "_current_frame->" +
                          file + "_progLen = metaOpenCLLoadProgramSource(\"" +
                          file + ".cl\", &__metacl_" + outFile +
                          "_current_frame->" + file + "_progSrc, &__metacl_" +
                          outFile + "_current_frame->" + file + "_progDir);\n";
    cache->runOnceInit += "    if (__metacl_" + outFile + "_current_frame->" +
                          file + "_progLen != -1)\n";
    cache->runOnceInit +=
        "      __metacl_" + outFile + "_current_frame->" + file +
        "_prog = clCreateProgramWithSource(__metacl_" + outFile +
        "_current_frame->context, 1, &__metacl_" + outFile +
        "_current_frame->" + file + "_progSrc, &__metacl_" + outFile +
        "_current_frame->" + file + "_progLen, &buildError);\n";
    cache->runOnceInit += "  }\n";
    cache->runOnceInit += "  if (__metacl_" + outFile + "_current_frame->" +
                          file + "_progLen != -1) {\n";
    cache->runOnceInit +=
        ERROR_CHECK("    ", "buildError", "OpenCL program creation error");
    cache->runOnceInit +=
        "    size_t args_sz = snprintf(NULL, 0, \"%s -I %s\", "
        "__metacl_" +
        file + "_custom_args ? __metacl_" + file +
        "_custom_args : \"\", __metacl_" + outFile + "_current_frame->" + file +
        "_progDir);\n";
    cache->runOnceInit += "    char * build_args = (char*) calloc(args_sz + 1, "
                          "sizeof(char));\n";
    cache->runOnceInit += "    snprintf(build_args, args_sz + 1, \"%s -I %s\", "
                          "__metacl_" +
                          file + "_custom_args ? __metacl_" + file +
                          "_custom_args : \"\", __metacl_" + outFile +
                          "_current_frame->" + file + "_progDir);\n";
    cache->runOnceInit += "    buildError = clBuildProgram(__metacl_" +
                          outFile + "_current_frame->" + file +
                          "_prog, 1, &__metacl_" + outFile +
                          "_current_frame->device, build_args, NULL, NULL);\n";
    cache->runOnceInit += "    free(build_args);\n";
    cache->runOnceInit += "    if (buildError != CL_SUCCESS) {\n";
    cache->runOnceInit += "      size_t logsize = 0;\n";
    cache->runOnceInit +=
        "      clGetProgramBuildInfo(__metacl_" + outFile + "_current_frame->" +
        file + "_prog, __metacl_" + outFile +
        "_current_frame->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);\n";
    cache->runOnceInit += "      char * buildLog = (char *) "
                          "malloc(sizeof(char) * (logsize + 1));\n";
    cache->runOnceInit += "      clGetProgramBuildInfo(__metacl_" + outFile +
                          "_current_frame->" + file + "_prog, __metacl_" +
                          outFile +
                          "_current_frame->device, CL_PROGRAM_BUILD_LOG, "
                          "logsize, buildLog, NULL);\n";
    cache->runOnceInit +=
        ERROR_CHECK("      ", "buildError", "OpenCL program build error");
    cache->runOnceInit +=
        "      fprintf(stderr, \"Build Log:\\n\%s\\n\", buildLog);\n";
    cache->runOnceInit += "      free(buildLog);\n";
    cache->runOnceInit += "    } else {\n";
    cache->runOnceInit += "      __metacl_" + outFile + "_current_frame->" +
                          file + "_init = 1;\n";
    cache->runOnceInit += "    }\n";
    // Moved      cache->runOnceInit += "  }\n";
    cache->runOnceDeinit += "      releaseError = clReleaseProgram(__metacl_" +
                            outFile + "_current_frame->" + file + "_prog);\n";
    cache->runOnceDeinit +=
        ERROR_CHECK("      ", "releaseError", "OpenCL program release error");
    cache->runOnceDeinit += "      free((char *)__metacl_" + outFile +
                            "_current_frame->" + file + "_progSrc);\n";
    cache->runOnceDeinit += "      free((char *)__metacl_" + outFile +
                            "_current_frame->" + file + "_progDir);\n";
    cache->runOnceDeinit += "      __metacl_" + outFile + "_current_frame->" +
                            file + "_progLen = 0;\n";
    cache->runOnceDeinit += "      __metacl_" + outFile + "_current_frame->" +
                            file + "_init = 0;\n";
    if (unified_output_c != NULL) {
      cache->outfile_c = unified_output_c;
      cache->outfile_h = unified_output_h;
    } else {
      std::error_code error_c, error_h;
      /// \todo FIXME: Check error returns;
      /// \todo FIXME Filter off the .cl extension before outfile creation
      cache->outfile_c =
          new llvm::raw_fd_ostream(file + ".c", error_c, llvm::sys::fs::F_None);
      cache->outfile_h =
          new llvm::raw_fd_ostream(file + ".h", error_h, llvm::sys::fs::F_None);
      llvm::errs() << error_c.message() << error_h.message();
    }

#if (__cplusplus >= 201402L)
    return std::make_unique<KernelASTConsumer>(&CI, cache->outfile_c,
                                                cache->outfile_h, file);
#else
    return llvm::make_unique<KernelASTConsumer>(&CI, cache->outfile_c,
                                                cache->outfile_h, file);
#endif
  }

private:
  CompilerInstance *CI;
};

/**
 * Finish generating all output files by organizing cached code from
 * AllHostCaches
 * \post All output .c/.h files are generated and flushed
 * \return An error code informing if anything went wrong
 * \todo TODO implement sane error codes to detect if anything went wrong
 */
int populateOutputFiles() {
  int errcode = 0;

  // booleans to streamline unified output checks and allow unification of code
  // below
  bool isUnified = (UnifiedOutputFile.getValue() != "");
  bool unifiedFirstPass = true;
  std::string outFileName;

  raw_ostream *out_c, *out_h;
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    // Get the output files
    if (isUnified) {
      out_c = unified_output_c;
      out_h = unified_output_h;
      outFileName = UnifiedOutputFile.getValue();
    } else {
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      // headers once per output
      *out_c << "//Force MetaMorph to include the OpenCL code\n";
      *out_c << "#ifdef __APPLE__\n";
      *out_c << "#include <OpenCL/opencl.h>\n";
      *out_c << "#else\n";
      *out_c << "#include <CL/opencl.h>\n";
      *out_c << "#endif\n";
      *out_c << "#include \"metamorph.h\"\n";
      *out_c << "#include \"metamorph_opencl.h\"\n";
      *out_c << "#include \"" + outFileName + ".h\"\n";

      // Emit user-defined types in the header file
      /// \todo TEST ensure we only get one copy of each in unified mode
      // We use the fileCachePair output .h file as the key since we may be in
      // unified mode but still want to get types from all input files
      for (std::pair<QualType, std::string> t :
           ImportedTypes[fileCachePair.second->outfile_h]) {
        *out_h << t.second;
      }
    }

    // Generate a space to place arguments (for each input file)
    *out_c << "//TODO: Expose this with a function (with safety checks) rather "
              "than a variable\n";
    *out_c << "const char * __metacl_" << fileCachePair.first
           << "_custom_args = NULL;\n";
    *out_h << "extern const char * __metacl_" << fileCachePair.first
           << "_custom_args;\n";
    // inhibit header block for the remaining files if unified
    unifiedFirstPass = false;
  }

  // Filling out the frame has to be done as a separate loop so that the unified
  // output correctly
  // accumulates across source files
  if (isUnified) {
    // Generate the unified module's OpenCL frame
    *out_h << "struct __metacl_" << outFileName << "_frame;\n";
    *out_h << "struct __metacl_" << outFileName << "_frame {\n";
    *out_h << "  struct __metacl_" << outFileName << "_frame * next_frame;\n";
    *out_h << "  cl_device_id device;\n  cl_context context;\n  "
              "cl_command_queue queue;\n";
  }
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      // Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
      // Generate the module's OpenCL frame
      *out_h << "struct __metacl_" << outFileName << "_frame;\n";
      *out_h << "struct __metacl_" << outFileName << "_frame {\n";
      *out_h << "  struct __metacl_" << outFileName << "frame * next_frame;\n";
      *out_h << "  cl_device_id device;\n  cl_context context;\n  "
                "cl_command_queue queue;\n";
    }
    // Generate the program variable (one per input file)
    /// \todo TODO support one-kernel-per-program convention?
    *out_h << "  const char * " << fileCachePair.first << "_progSrc;\n";
    *out_h << "  size_t " << fileCachePair.first << "_progLen;\n";
    *out_h << "  const char * " << fileCachePair.first << "_progDir;\n";
    *out_h << "  cl_program " << fileCachePair.first << "_prog;\n";
    *out_h << "  cl_int " << fileCachePair.first << "_init;\n";
    // Add the kernel variables
    for (std::string var : fileCachePair.second->cl_kernels) {
      *out_h << var;
    }
  }

  // reset for this loop
  unifiedFirstPass = true;
  // Loop to close up the frames and *begin* initializer implementation
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      // Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      // Finish the struct (in the header) and create a pointer to this module's
      // active copy
      *out_h << "};\n";
      *out_c << "struct __metacl_" << outFileName << "_frame * __metacl_"
             << outFileName << "_current_frame = NULL;\n";
      *out_c << "\n";
      // Generate a lookup function to get an initialized frame for the module
      // based on the queue Should this be exposed to the user?
      *out_h << "struct __metacl_" << outFileName << "_frame * __metacl_"
             << outFileName << "_lookup_frame(cl_command_queue queue);\n";
      *out_c << "struct __metacl_" << outFileName << "_frame * __metacl_"
             << outFileName << "_lookup_frame(cl_command_queue queue) {\n";
      *out_c << "  struct __metacl_" << outFileName
             << "_frame * current = __metacl_" << outFileName
             << "_current_frame;\n";
      *out_c << "  while (current != NULL) {\n";
      *out_c << "    if (current->queue == queue) break;\n";
      *out_c << "    current = current->next_frame;\n";
      *out_c << "  }\n";
      *out_c << "  return current;\n";
      *out_c << "}\n";

      // Generate the MetaMorph registration function
      *out_h << "#ifdef __cplusplus\n";
      *out_h << "extern \"C\" {\n";
      *out_h << "#endif\n";
      *out_h << "meta_module_record * metacl_" << outFileName
             << "_registry(meta_module_record * record);\n";
      *out_c << "meta_module_record * metacl_" << outFileName
             << "_registration = NULL;\n";
      *out_c << "meta_module_record * metacl_" << outFileName
             << "_registry(meta_module_record * record) {\n";
      *out_c << "  if (record == NULL) return metacl_" << outFileName
             << "_registration;\n";
      *out_c << "  meta_module_record * old_registration = metacl_"
             << outFileName << "_registration;\n";
      *out_c << "  if (old_registration == NULL) {\n";
      *out_c << "    record->implements = module_implements_opencl;\n";
      *out_c << "    record->module_init = &metacl_" << outFileName
             << "_init;\n";
      *out_c << "    record->module_deinit = &metacl_" << outFileName
             << "_deinit;\n";
      *out_c << "    record->module_registry_func = &metacl_" << outFileName
             << "_registry;\n";
      *out_c << "    metacl_" << outFileName << "_registration = record;\n";
      *out_c << "  }\n";
      *out_c << "  if (old_registration != NULL && old_registration != record) "
                "return record;\n";
      *out_c << "  if (old_registration == record) metacl_" << outFileName
             << "_registration = NULL;\n";
      *out_c << "  return old_registration;\n";
      *out_c << "}\n";
      // Generate the initialization wrapper
      *out_h << "void metacl_" << outFileName << "_init();\n";
      *out_c << "void metacl_" << outFileName << "_init() {\n";
      *out_c << "  cl_int buildError, createError;\n";
      // Ensure the module is registered
      *out_c << "  if (metacl_" << outFileName << "_registration == NULL) {\n";
      *out_c << "    meta_register_module(&metacl_" << outFileName
             << "_registry);\n";
      *out_c << "    return;\n";
      *out_c << "  }\n";
      // Ensure a program/kernel storage frame is initialized
      *out_c << "  struct __metacl_" << outFileName
             << "_frame * new_frame = (struct __metacl_" << outFileName
             << "_frame *) calloc(1, sizeof(struct __metacl_" << outFileName
             << "_frame));\n";
      *out_c << "  new_frame->next_frame = __metacl_" << outFileName
             << "_current_frame;\n";
      // Fill in the new frame directly from MetaMorph, then make sure it exists
      *out_c << "  meta_get_state_OpenCL(NULL, &new_frame->device, "
                "&new_frame->context, &new_frame->queue);\n";
      // Ensure a MetaMorph OpenCL state exists
      *out_c << "  if (new_frame->context == NULL) {\n";
      *out_c << "    metaOpenCLFallback();\n";
      *out_c << "    meta_get_state_OpenCL(NULL, &new_frame->device, "
                "&new_frame->context, &new_frame->queue);\n";
      *out_c << "  }\n";
      *out_c << "  __metacl_" << outFileName << "_current_frame = new_frame;\n";
      *out_c << "  meta_cl_device_vendor vendor = "
                "metaOpenCLDetectDevice(new_frame->device);\n";
    }
    // Add the clCreateProgram bits
    *out_c << fileCachePair.second->runOnceInit;
    // Add the clCreateKernel bits
    for (std::string kern : fileCachePair.second->kernelInit) {
      *out_c << kern;
    }
    // Finish the program's creation block
    *out_c << "  }\n";
    // inhibit registration on subsequent passes in unified mode
    unifiedFirstPass = false;
  }

  // reset for this loop
  unifiedFirstPass = true;
  // Finishing the initializers and starting the deinitializers has to be
  // separate so the unified mode can accumulate across inputs
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      // Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      *out_c << "  metacl_" << outFileName
             << "_registration->initialized = 1;\n";
      *out_c << "}\n\n";
      // Generate the deconstruction wrapper
      *out_h << "void metacl_" << outFileName << "_deinit();\n";
      *out_c << "void metacl_" << outFileName << "_deinit() {\n";
      *out_c << "  cl_int releaseError;\n";
      // Ensure we are deregistered with MetaMorph-core
      //      *out_c << "  if (metacl_" << outFileName << "_registration !=
      //      NULL) {\n"; *out_c << "    meta_deregister_module(&metacl_" <<
      //      outFileName << "_registry);\n"; *out_c << "    return;\n"; *out_c
      //      << "  }\n";
      // Esnure a program/kernel storage frame exists
      *out_c << "  if (__metacl_" << outFileName
             << "_current_frame != NULL) {\n";
    }
    // Release all the kernels
    *out_c << "    if (__metacl_" << outFileName << "_current_frame->"
           << fileCachePair.first << "_progLen != -1) {\n";
    for (std::string kern : fileCachePair.second->kernelDeinit) {
      *out_c << kern;
    }
    *out_c << fileCachePair.second->runOnceDeinit;
    *out_c << "    }\n";

    // inhibit deinit generation on subsequent unified passes
    unifiedFirstPass = false;
  }

  // Reset for this loop
  unifiedFirstPass = true;
  // Free and finalize has to be a separate loop so unified mode can accumulate
  // Releases correctly
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      // Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      // Release the program and frame
      *out_c << "    struct __metacl_" << outFileName
             << "_frame * next_frame = __metacl_" << outFileName
             << "_current_frame->next_frame;\n";
      *out_c << "    free(__metacl_" << outFileName << "_current_frame);\n";
      *out_c << "    __metacl_" << outFileName
             << "_current_frame = next_frame;\n";
      *out_c << "    if (__metacl_" << outFileName
             << "_current_frame == NULL && metacl_" << outFileName
             << "_registration != NULL) {\n";
      *out_c << "      metacl_" << outFileName
             << "_registration->initialized = 0;\n";
      *out_c << "    }\n";
      *out_c << "  }\n";
      // Finish the deinit wrapper
      *out_c << "}\n\n";
    }

    // Add the kernel wrappers themselves
    for (std::string proto : fileCachePair.second->hostProtos) {
      *out_h << proto;
    }
    for (std::string impl : fileCachePair.second->hostFuncImpls) {
      *out_c << impl;
    }
    if (!isUnified) {
      out_c->flush();
      *out_h << "#ifdef __cplusplus\n";
      *out_h << "}\n";
      *out_h << "#endif\n";
      out_h->flush();
    }
    // Inhibit frame release on subsequent unified passes
    unifiedFirstPass = false;
  }
  // Finalize separately so all wrappers are correctly accumulated
  if (isUnified) {
    out_c->flush();
    *out_h << "#ifdef __cplusplus\n";
    *out_h << "}\n";
    *out_h << "#endif\n";
    out_h->flush();
  }

  return errcode;
}

/**
 * Simple driver to bootstrap the Clang machinery that reads the input .cl
 * files, and then populates output files with generated code
 * \param argc The number of command line tokens
 * \param argv a vector of C strings of all command-line tokens
 * \return an error code to return to the OS informing whether anything went
 * wrong
 * \todo TODO Implement sane error codes
 * \post all input .cl files have been read andd appropriate host
 * implementations and header files generated
 */
int main(int argc, const char **argv) {
  int errcode = 0;
  CommonOptionsParser op(argc, argv, MetaCLCategory);
  // If they want unified output, generate the files
  if (UnifiedOutputFile.getValue() != "") {
    std::error_code error;
    /// \todo FIXME Check error returns
    unified_output_c = new llvm::raw_fd_ostream(
        UnifiedOutputFile.getValue() + ".c", error, llvm::sys::fs::F_None);
    unified_output_h = new llvm::raw_fd_ostream(
        UnifiedOutputFile.getValue() + ".h", error, llvm::sys::fs::F_None);
  }
  // If they want optional or disabled MetaMorph integration, create the headers
  // and shim
  if (UseMetaMorph.getValue() &
      metaMorphDisabled) { // By explicitly assigning the enum like a
                           // bitfield,both disabled and optional will evaluate
                           // true
    std::error_code error;
    raw_ostream *metamorph_h =
        new llvm::raw_fd_ostream("metamorph.h", error, llvm::sys::fs::F_None);
    raw_ostream *metamorph_opencl_h = new llvm::raw_fd_ostream(
        "metamorph_opencl.h", error, llvm::sys::fs::F_None);
    raw_ostream *metamorph_shim_c = new llvm::raw_fd_ostream(
        "metamorph_shim.c", error, llvm::sys::fs::F_None);
    // If the support level is optional, inject the defines necessary to do the
    // dynamic loading and binding
    if (UseMetaMorph.getValue() == metaMorphOptional) {
      // Do injection
      metamorph_shim_c->write(_binary_shim_dynamic_h_start,
                              _binary_shim_dynamic_h_end -
                                  _binary_shim_dynamic_h_start);
      *metamorph_shim_c << "\n";
    }
    // Add the raw text from the linked objects directly to the output streams
    metamorph_h->write(_binary_metamorph_emulatable_h_start,
                       _binary_metamorph_emulatable_h_end -
                           _binary_metamorph_emulatable_h_start);
    metamorph_opencl_h->write(_binary_metamorph_opencl_emulatable_h_start,
                              _binary_metamorph_opencl_emulatable_h_end -
                                  _binary_metamorph_opencl_emulatable_h_start);
    metamorph_shim_c->write(_binary_metamorph_shim_c_start,
                            _binary_metamorph_shim_c_end -
                                _binary_metamorph_shim_c_start);
    metamorph_h->flush();
    metamorph_opencl_h->flush();
    metamorph_shim_c->flush();
  }
  CompilationDatabase &CompDB = op.getCompilations();
  ClangTool Tool(CompDB, op.getSourcePathList());
  Tool.run(newFrontendActionFactory<MetaCLFrontendAction>().get());

  // After the tool runs, dump all the host code we have cached to appropriate
  // output files.
  errcode = populateOutputFiles();
  return errcode;
}
