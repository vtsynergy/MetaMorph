/**
* (c) 2018 Virginia Tech
*
* Please see license information in the main MetaMorph repository
*
* MetaGen-CL
* A tool to consume OpenCL kernel files and produce MetaMorph-compatible
* host-side wrappers for the contained kernels.
*
* ALPHA/Prototype software, no warranty expressed or implied.
*
* Authors: Paul Sathre
*
*/

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/Attr.h" //This file has the auto-generated implementations of all the attribute classes, reference it if you need to get information out of an attribute

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

//Ternary checks the status of the inline-error-check variable to easily eliminate all checks added through the macro
#define ERROR_CHECK(errorcode, text) (InlineErrorCheck.getValue() ? "  if (" errorcode " != CL_SUCCESS) fprintf(stderr, \"" text " \%d at \%s:\%d\\n\", " errorcode ", __FILE__, __LINE__);\n" : "")
#define APPEND_ERROR_CHECK(string, errorcode, text) string += ERROR_CHECK(errorcode, text)

static llvm::cl::OptionCategory MetaGenCLCategory("MetaGen-CL Options");

llvm::cl::opt<std::string, false> UnifiedOutputFile("unified-output-file", llvm::cl::desc("If a filename is provided, all kernel files will generate a single set of host wrappers, instead of one per file."), llvm::cl::value_desc("<\"filename\">"), llvm::cl::init(""));
llvm::cl::opt<bool, false> InlineErrorCheck("inline-error-check", llvm::cl::desc("Generate an immediate error check after every OpenCL runtime call."), llvm::cl::value_desc("<true/false>"), llvm::cl::init(true));
llvm::cl::opt<bool, false> OverwriteFiles("overwrite-files", llvm::cl::desc("Instead of trying to replace only MetaCL-generated code in existing output files, simply overwrite them in entirety."), llvm::cl::value_desc("<true/false>"), llvm::cl::init(false));
raw_ostream * unified_output_c = NULL, * unified_output_h = NULL;

//Meant to store all the boilerplate from a single input kernel file
typedef struct {
  //clCreateProgram, clBuildProgram, cl_programs, diagnostics
  std::string runOnceInit;
  //clReleaseProgram
  std::string runOnceDeinit;
  //All the global cl_kernel declarations for an input file
  std::vector<std::string> cl_kernels;
  //clCreateKernel
  std::vector<std::string> kernelInit;
  //clReleaseKernel
  std::vector<std::string> kernelDeinit;
  //prototypes of host-side kernel wrappers, indexed by name, with arg string as the value
  std::vector<std::string> hostProtos;
  //Implementations of host-side kernel wrappers (w/o prototypes), indexed by host-side name, with function contents as value
  std::vector<std::string> hostFuncImpls;
  raw_ostream * outfile_h;
  raw_ostream * outfile_c;
} hostCodeCache;

//Map of all hostCodeCaches (indexed by input filename)
std::map<std::string, hostCodeCache *> AllHostCaches;

class PrototypeHandler : public MatchFinder::MatchCallback {
  public:
    PrototypeHandler(raw_ostream * c_outfile, raw_ostream * h_outfile, std::string infile) : output_file_c(c_outfile), output_file_h(h_outfile), filename(infile) {
      //Any initialization?
    }
    virtual void run(const MatchFinder::MatchResult &Result);

  typedef struct  {
    QualType devType;
    std::string hostType;
    unsigned int devPos;
    //TODO add other flags (restrict, addrspace, etc)
    //if it is local memory we need the size, not a pointer
    bool isLocal = false;
    //if it is global or const, we need a pointer (cl_mem)
    bool isGlobalOrConst = false;
    std::string name;
  } argInfo;

  private:
    std::string filename = "";
    raw_ostream * output_file_c = NULL;
    raw_ostream * output_file_h = NULL;
};


//FIXME this sort was arbitrarily chosen, replace it with something else if it allows types to be used before defined
namespace std {
template<> struct less<QualType> {
 bool operator()(const QualType first, const QualType second) {
  return (first.getAsOpaquePtr() < second.getAsOpaquePtr());
}
};
}

//For a given Qualtype will do a postorder traversal of all required records and typedefs to ensure they are generated in the given output header file
//Returns the host-side name of the imported type
std::map<raw_ostream *, std::map<QualType, std::string> > ImportedTypes;
std::string getHostType(QualType type, ASTContext &Context, raw_ostream * out_header);

std::string importTypeDependencies(QualType type, ASTContext &Context, raw_ostream * out_header, bool alreadyImported = false) {
  //If it's a primitive type, don't store anything but convert it to the OpenCL type
  if (isa<BuiltinType>(type)) {
    return getHostType(type, Context, out_header);
  }

  //If it's an OpenCL "primitive" type with embedded typedef (like uchar --> unsigned char) don't import it
  //We can recognize these by their presence in Clang's opencl-c.h file
  if (const TypedefType * t = dyn_cast<TypedefType>(type)) {
    if (Context.getSourceManager().getBufferName(t->getDecl()->getLocation()).str().find("opencl-c.h") != std::string::npos) {
      return getHostType(type, Context, out_header);
    }
  }

  //If it's an array type, make sure we resolve the element types dependencies/rewrites
  if (const ArrayType * a = dyn_cast<ArrayType>(type)) {
    std::string imported = importTypeDependencies(a->getElementType(), Context, out_header);
    //to ensure the type is rewritten properly we must reconstruct the array type after import
    std::string type_str = type.getAsString();
    std::string elem_str = a->getElementType().getAsString();
    //if the imported type is an array, we don't need the extra size tags for modifying our own return type
    std::string::size_type pos;
    if ((pos = elem_str.find(' ')) != std::string::npos) elem_str.erase(pos);
    if ((pos = imported.find(' ')) != std::string::npos) imported.erase(pos);
    type_str.replace(type_str.find(elem_str), elem_str.length(), imported);
    return type_str;
  }

  //If the specific type we need is not constructed yet
  if (ImportedTypes[out_header].find(type) == ImportedTypes[out_header].end()) {
    //We have to record that we've imported it before we recurse so that it acts as it's own base case (else it'll recurse infinitely)
    (ImportedTypes[out_header])[type] = "";
    //Remember if an outer typdef is responsible for defining an interior RecordType
    bool firstEncountered = false;
    //If this is a combined Typedef/Record decl, the internal recursions for the ElaboratedType and RecordType do not need to re-import the text;
    if (isa<TypedefType>(type) && isa<ElaboratedType>(type.getSingleStepDesugaredType(Context))) {
    //If this is the first time the internal elaborated struct is encountered, remember that we need to implement it ourselves because it will be inhibited
      firstEncountered = (ImportedTypes[out_header].find(type.getSingleStepDesugaredType(Context)) == ImportedTypes[out_header].end());
      importTypeDependencies(type.getSingleStepDesugaredType(Context), Context, out_header, true);
    } else if (isa<ElaboratedType>(type) && isa<RecordType>(type.getSingleStepDesugaredType(Context))) {
      importTypeDependencies(type.getSingleStepDesugaredType(Context), Context, out_header, true);
    } else {
      //Remember the outer recursive instance's import status
      importTypeDependencies(type.getSingleStepDesugaredType(Context), Context, out_header, alreadyImported);
    }
    if (const TypedefType * t = dyn_cast<TypedefType>(type)) {
      if (!alreadyImported) {
        //This elaborate if simply sorts out records that would be anonymous if not for a surrounding typdef, which must be handled differently
        //TODO: support unions and enums, any types of records other than struct (Found in TagDecl)
        if (isa<ElaboratedType>(t->getDecl()->getUnderlyingType()) && isa<RecordType>(type.getSingleStepDesugaredType(Context).getSingleStepDesugaredType(Context)) && dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context).getSingleStepDesugaredType(Context))->getDecl()->getTypedefNameForAnonDecl() != NULL) {
          std::string structName = dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context).getSingleStepDesugaredType(Context))->getDecl()->getTypedefNameForAnonDecl()->getName();
          //This first branch deals with a singular typedef that defines multiple types pointing to the same anonymous struct, all but the first will be generated as separate decls
          if (structName != t->getDecl()->getName()) {
            std::string record = "typedef " + structName + " " + t->getDecl()->getName().str() + ";\n";
	    (ImportedTypes[out_header])[type] = record;
          } else {
          //This branch handles the primary (which inherits its name from the first typedef'd type
	    //construct the record in-place
	    std::string record = "typedef struct {\n";
	    for (const FieldDecl * field : dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context).getSingleStepDesugaredType(Context))->getDecl()->fields()) {
              if (isa<ArrayType>(field->getType())) {
                std::string arr_type = importTypeDependencies(field->getType(), Context, out_header);
                
                record += "  " + arr_type.insert(arr_type.find("["), field->getName().str()) +";\n";
              } else {
                record += "  " + importTypeDependencies(field->getType(), Context, out_header) + " " + field->getName().str() + ";\n";
              }
            }
            record += "} " + type.getAsString() + ";\n";
            (ImportedTypes[out_header])[type] = record;
          }
        } else {
          //Typedefs of named types (either scalars or records)
         //If the underlying type is a record and hasn't been constructed yet, it will be inhibited from generating itself so we have to handle it here
         if (isa<ElaboratedType>(t->getDecl()->getUnderlyingType()) && firstEncountered) {
            std::string record = "typedef " + t->getDecl()->getUnderlyingType().getAsString() + " {\n";
	    for (const FieldDecl * field : dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context).getSingleStepDesugaredType(Context))->getDecl()->fields()) {
              if (isa<ArrayType>(field->getType())) {
                std::string arr_type = importTypeDependencies(field->getType(), Context, out_header);
                
                record += "  " + arr_type.insert(arr_type.find("["), field->getName().str()) +";\n";
              } else {
                record += "  " + importTypeDependencies(field->getType(), Context, out_header) + " " + field->getName().str() + ";\n";
              }
            }
            (ImportedTypes[out_header])[type] = record + "} " + type.getAsString() + ";\n";
           
          } else {
            (ImportedTypes[out_header])[type] = "typedef " + importTypeDependencies(t->getDecl()->getUnderlyingType(), Context, out_header) + " " + type.getAsString() + ";\n";
          }
        }
      }
    }
    //If it's an ElaboratedType without a wrapping typedef
    if (const ElaboratedType * e = dyn_cast<ElaboratedType>(type)) {
      if (!alreadyImported) {
        //construct the record in-place
        std::string record = type.getAsString() + " {\n";
        for (const FieldDecl * field : dyn_cast<RecordType>(type.getSingleStepDesugaredType(Context))->getDecl()->fields()) {
          record += "  " + importTypeDependencies(field->getType(), Context, out_header) + " " + field->getName().str() + ";\n";
        }
        record += "};\n";
        (ImportedTypes[out_header])[type] = record;
      }
    }
    //If it's a Record without an explicit keyword (like struct or union)
    if (const RecordType * r = dyn_cast<RecordType>(type)) {
      if (!alreadyImported) {
        std::string record = type.getAsString() + " {\n";
        for (const FieldDecl * field : r->getDecl()->fields()) {
          record += "  " + importTypeDependencies(field->getType(), Context, out_header) + " " + field->getName().str() + ";\n";
        }
        record += "};\n";
        (ImportedTypes[out_header])[type] = record;
      }
    }
  }
  //Anything that isn't caught by the builtin, OpenCL, and array filters is going to preserve it's own name
  return type.getAsString();
}
std::string getHostType(QualType devType, ASTContext &Context, raw_ostream * stream) {
  std::string retType = "";
  //TODO handle types that don't directly map to a cl_type (images, vectors)
  retType += "cl_";
  std::string canon = devType.getAsString();
  std::string::size_type pos;
  if ((pos = canon.find("unsigned ")) != std::string::npos) canon.replace(pos, 9, "u"); 
  //strip off and appropriately modify vectorization attributes "__attribute__((ext_vector_type(<width>)))
  //Assumes the intervening pointer type has already been removed
  if ((pos = canon.find(" __attribute__((ext_vector_type(")) != std::string::npos) {
    std::string::size_type endPos = canon.find(")))", pos);
    canon.erase(endPos, 3);
    canon.erase(pos, 32);
  }

  //FIXME Technically Clang should catch bool struct elements (and it does if it's directly a parameter, but not if the param is a pointer to a struct with a bool in it)
  if (canon == "_Bool") return "\n#error passing a boolean through a pointer or struct pointer in OpenCL is undefined\ncl_" + canon;
  
  return "cl_" + canon;
}

PrototypeHandler::argInfo * analyzeDevParam(ParmVarDecl * devParam, ASTContext &Context, raw_ostream * stream) {
  PrototypeHandler::argInfo * retArg = new PrototypeHandler::argInfo();
  //Detect the type of the device parameter
  retArg->devType = devParam->getType();
  //If it is a pointer type, add address space flags
  //FIXME Deal with NULL TypePtr
  if (retArg->devType->isAnyPointerType()) {
    clang::LangAS addrSpace = retArg->devType->getPointeeType().getAddressSpace();
    if (addrSpace == LangAS::opencl_local) retArg->isLocal = true;
    if (addrSpace == LangAS::opencl_global || addrSpace == LangAS::opencl_constant) retArg->isGlobalOrConst = true;
    retArg->hostType = importTypeDependencies(retArg->devType->getPointeeType().getUnqualifiedType(), Context, stream);
  //retArg->hostType = getHostType(retArg->devType, Context, stream);
  } else {
    retArg->hostType = importTypeDependencies(retArg->devType.getUnqualifiedType(), Context, stream);
  }
  //borrow the name
  retArg->name = devParam->getNameAsString();
  //TODO grab any additional flags we need to keep in mind (restrict, constant, address space etc)
  return retArg;
}

std::string trimFileSlashesAndType(std::string in) {
      size_t dotPos = in.rfind('.');
      size_t slashPos = in.rfind('/') + 1;
	//TODO handle missing dot or slash
      return in.substr(slashPos, dotPos-slashPos);
}

void PrototypeHandler::run(const MatchFinder::MatchResult &Result) {
  bool singleWorkItem = false;
  const FunctionDecl * func;
  const FunctionDecl * nd_func = NULL;
  std::string outFile = ((UnifiedOutputFile.getValue() == "") ? filename : UnifiedOutputFile.getValue());
  if (func = Result.Nodes.getNodeAs<FunctionDecl>("swi_kernel_def")) {
    singleWorkItem = true;
    nd_func = Result.Nodes.getNodeAs<FunctionDecl>("nd_func");
  } else func = Result.Nodes.getNodeAs<FunctionDecl>("kernel_def");
  if (func) {
    //Figure out the host-side name
    //TODO If the name of the kernel ends in "_<cl_type>", strip it and register it as an explicitly typed kernel (for sharing a host wrapper)
    //TODO for all other cases, create a new meta_typeless type
      std::string host_func = "meta_gen_opencl_" + filename + "_" + func->getNameAsString();
    //TODO use the work_group_size information
    //TODO hoist attribute handliong out to it's own function that returns a single data structure with all the kernel attributes we might need to handle
    unsigned int work_group_size[4] = {1, 1, 1, 0}; //4th member is the type of the size (0 = unspecified, 1 = hint, 2 = required, 3 = intel required)
    for (auto attr : func->getAttrs()) {
      //TODO Xilinx adds the xcl_max_work_group_size and xcl_zero_global_work_offset attributes to kernels
      //TODO clang doesn't appear to have the OpenCL pointer nosvm attribute yet
      //TODO recognize the endian attribute so that if it is explicitly specified, we can warn in the host API
      //TODO implement handlers for any necessary kernel type attributes
      if (VecTypeHintAttr * vecAttr = dyn_cast<VecTypeHintAttr>(attr)) {
        //TODO do something with it
        vecAttr->getTypeHint().getAsString();
      } else if (WorkGroupSizeHintAttr * sizeAttr = dyn_cast<WorkGroupSizeHintAttr>(attr)) {
        if (work_group_size[3] < 2) { //Only if we don't already have a required size
          work_group_size[0] = sizeAttr->getXDim(), work_group_size[1] = sizeAttr->getYDim(), work_group_size[2] = sizeAttr->getZDim(), work_group_size[3] =1;
        }
        llvm::errs() << "Suggested work group size is (" << work_group_size[0] << ", " << work_group_size[1] << ", " << work_group_size[2] << ")\n";
      } else if (ReqdWorkGroupSizeAttr * sizeAttr = dyn_cast<ReqdWorkGroupSizeAttr>(attr)) {
        if (work_group_size[3] < 2) { //Only if we don't already have a required size
          work_group_size[0] = sizeAttr->getXDim(), work_group_size[1] = sizeAttr->getYDim(), work_group_size[2] = sizeAttr->getZDim(), work_group_size[3] =2;
        }
        llvm::errs() << "Required work group size is (" << work_group_size[0] << ", " << work_group_size[1] << ", " << work_group_size[2] << ")\n";
      } else if (OpenCLIntelReqdSubGroupSizeAttr * subSize = dyn_cast<OpenCLIntelReqdSubGroupSizeAttr>(attr)) {
        //TODO Do we need to handle it?
      } //TODO other important kernel attributes
    }
    //TODO implement asynchronous calls (allow it to wait on kernels as well as return event type)
    //TODO check the prototype for any OpenCL-specific attributes that require the host to behave in a particular way
    
    //Creating the AST consumer forces all the once-per-input boilerplate to be generated so we don't have to do it here
        hostCodeCache * cache = AllHostCaches[filename];
        std::string framed_kernel = "__meta_gen_opencl_" + outFile + "_current_frame->" + func->getNameAsString() + "_kernel";
	cache->cl_kernels.push_back("  cl_kernel " + func->getNameAsString() + "_kernel;\n");
    //Generate a clCreatKernelExpression
	cache->kernelInit.push_back("    " + framed_kernel + " = clCreateKernel(__meta_gen_opencl_" + outFile + "_current_frame->" + filename + "_prog, \"" + func->getNameAsString() + "\", &createError);\n");
        cache->kernelInit.push_back(ERROR_CHECK("createError", "OpenCL kernel creation error"));
    //Generate a clReleaseKernelExpression
	cache->kernelDeinit.push_back("    releaseError = clReleaseKernel(" + framed_kernel + ");\n");
        cache->kernelDeinit.push_back(ERROR_CHECK("releaseError", "OpenCL kernel release error"));

	//Begin constructing the host wrapper
	std::string hostProto = "cl_int " + host_func + "(";
	std::string setArgs = "";
        std::string doxygen = "/** Automatically-generated by MetaGen-CL\n";
        if (singleWorkItem) doxygen += "Kernel function is detected as Single-Work-Item\n";
    doxygen += "\\param grid_size a size_t[3] providing the number of workgroups in the X and Y dimensions, and the number of iterations in the Z dimension\n";
    doxygen +="\\param block_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions";
    if (singleWorkItem) {
      doxygen += " (detected as single-work-item, must be {1, 1, 1})\n";
    } else if (work_group_size[3] == 1) {
      doxygen += " (work_group_size_hint attribute suggests {" + std::to_string(work_group_size[0]) + ", " + std::to_string(work_group_size[1]) + ", " + std::to_string(work_group_size[2]) + "})\n";
    } else if (work_group_size[3] == 2) {
      doxygen += " (reqd_work_group_size attribute requires {" + std::to_string(work_group_size[0]) + ", " + std::to_string(work_group_size[1]) + ", " + std::to_string(work_group_size[2]) + "})\n";
    } else {
      doxygen += "\n";
    }

    //Add module-registration check/lazy registration
    setArgs += "  if (meta_gen_opencl_" + outFile + "_registration == NULL) meta_register_module(&meta_gen_opencl_" + outFile + "_registry);\n";

        //TODO Add other wrapper fuction parameters
    hostProto += "size_t (*grid_size)[3], size_t (*block_size)[3], ";
    setArgs += "  cl_int retCode = CL_SUCCESS;\n";
    //Add pseudo auto-scaling safety code
    setArgs += "  size_t grid[3];\n";
    if (work_group_size[3] == 0 && !singleWorkItem) {
      setArgs += "  size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;\n";
    } else if (singleWorkItem) {
      setArgs += "  size_t block[3] = {1, 1, 1};\n";
    } else {
      setArgs += "  size_t block[3] = {" + std::to_string(work_group_size[0]) + ", " + std::to_string(work_group_size[1]) + ", " + std::to_string(work_group_size[2]) + "};\n";
    }
    setArgs += "  int iters;\n\n";
    setArgs += "  //Default runs a single workgroup\n";
    setArgs += "  if (grid_size == NULL || block_size == NULL) {\n";
    setArgs += "    grid[0] = block[0];\n    grid[1] = block[1];\n    grid[2] = block[2];\n";
    setArgs += "    iters = 1;\n";
    setArgs += "  } else {\n";
    if (work_group_size[3] == 1) {
      setArgs += "    if (!(block[0] == block_size[0] && block[1] == block_size[1] && block[2] == block_size[2])) {\n";
      setArgs += "      fprintf(stderr, \"Warning: kernel " + func->getNameAsString() + " suggests a workgroup size of {" + std::to_string(work_group_size[0]) + ", " + std::to_string(work_group_size[1]) + ", " + std::to_string(work_group_size[2]) + "} at \%s:\%d\\n\", __FILE__, __LINE__);\n";
      setArgs += "    }\n";

    } else if (work_group_size[3] == 2 || singleWorkItem) {
      setArgs += "    if (!(block[0] == block_size[0] && block[1] == block_size[1] && block[2] == block_size[2])";
      if (singleWorkItem) setArgs += " && !(grid[0] == 1 && grid[1] == 1 && grid[2] == 1)";
      setArgs += ") {\n";
      setArgs += "      fprintf(stderr, \"Error: kernel " + func->getNameAsString() + " requires a workgroup size of {" + std::to_string(work_group_size[0]) + ", " + std::to_string(work_group_size[1]) + ", " + std::to_string(work_group_size[2]) + "}, aborting launch at \%s:\%d\\n\", __FILE__, __LINE__);\n";
      setArgs += "      return CL_INVALID_WORK_GROUP_SIZE;\n";
      setArgs += "    }\n";
    }
    setArgs += "    grid[0] = (*grid_size)[0] * (*block_size)[0];\n";
    setArgs += "    grid[1] = (*grid_size)[1] * (*block_size)[1];\n";
    setArgs += "    grid[2] = (*block_size)[2];\n";
    setArgs += "    block[0] = (*block_size)[0];\n";
    setArgs += "    block[1] = (*block_size)[1];\n";
    setArgs += "    block[2] = (*block_size)[2];\n";
    setArgs += "    iters = (*grid_size)[2];\n";
    setArgs += "  }\n";

    int pos = 0;
    for (ParmVarDecl * parm : func->parameters()) {
      //Figure out the host-side representation of the param
      PrototypeHandler::argInfo * info = analyzeDevParam(parm, *Result.Context, output_file_h);
      //Add it to the wrapper param list
      //If it's not local, directly use the host type and get/set the data itself
      if (info->isGlobalOrConst) {
        hostProto += "cl_mem * " + info->name + ", ";
        setArgs += "  retCode = clSetKernelArg(" + framed_kernel + ", " + std::to_string(pos) + ", sizeof(cl_mem), " + info->name + ");\n";
        setArgs += ERROR_CHECK("retCode", "OpenCL kernel argument assignment error (arg: \\\"" + info->name + "\\\", host wrapper: \\\"" + host_func + "\\\")");
        doxygen += "\\param " + info->name + " a cl_mem buffer, must internally store " + info->hostType + " types\n";
      } else if (info->isLocal) { //If it is local, instead create a size variable and set the size of the memory region
        hostProto += "size_t " + info->name + "_num_local_elems, ";
        setArgs += "  retCode = clSetKernelArg(" + framed_kernel + ", " + std::to_string(pos) + ", sizeof(" + info->hostType + ") * " + info->name + "_num_local_elems, NULL);\n";
        setArgs += ERROR_CHECK("retCode", "OpenCL kernel argument assignment error (arg: \\\"" + info->name + "\\\", host wrapper: \\\"" + host_func + "\\\")");
        doxygen += "\\param " + info->name + "_num_local_elems allocate __local memory space for this many " + info->hostType + " elements\n";
      } else {
        hostProto += info->hostType + " " + info->name + ", ";
        //generate a clSetKernelArg expression
        setArgs += "  retCode = clSetKernelArg(" + framed_kernel + ", " + std::to_string(pos) + ", sizeof(" + info->hostType + "), &" + info->name + ");\n";
        setArgs += ERROR_CHECK("retCode", "OpenCL kernel argument assignment error (arg: \\\"" + info->name + "\\\", host wrapper: \\\"" + host_func + "\\\")");
        doxygen += "\\param " + info->name + " scalar parameter of type \"" + info->hostType + "\"\n";
      }
      pos++;
    }
    //Remove the extra ", "
    hostProto.erase(hostProto.size()-2, 2);

        //TODO Add other wrapper function parameters
    //FIXME For now the MM type is ignored TODO add a a_typeless type to metamorph's type enum
//    hostProto += ", meta_type_id type, int async, cl_event * event";
    hostProto += ", int async, cl_event * event";
//    doxygen += "\\param type the MetaMorph type of the function\n";
    doxygen += "\\param async whether the kernel should run asynchronously\n";
    doxygen += "\\param event returns the cl_event corresponding to the kernel launch if run asynchronously\n";
    //Add the forward declaration now that it's finished
    cache->hostProtos.push_back(hostProto + ");\n");

    //TODO can we detect work size based on calls to get_global_id, get_local_id, etc?
    //TODO add enforced host restrictions to doxygen
    doxygen += "*/\n";

    //Assemble the wrapper definition
    std::string wrapper = "";
    wrapper += doxygen + hostProto + ") {\n";
    //TODO add other initialization
    //Add the clSetKernelArg calls;
    wrapper += setArgs;
    //TODO autodetect work dimensions
    int workDim = 1;
    //TODO handle work offset
    std::string offset = "NULL";
    //TODO handle worksizes
    std::string globalSize = "grid", localSize = "block";
    //TODO handle events
    std::string eventWaitListSize = "0", eventWaitList = "NULL", retEvent = "event";
    //Add the launch
    //TODO Detect if single-work-item from kernel funcs (not just attribute)
    if (singleWorkItem || (work_group_size[3] != 0 && work_group_size[0] == 1 && work_group_size[1] == 1 && work_group_size[2] == 1)) {
      wrapper += "  retCode = clEnqueueTask(meta_queue, " + framed_kernel + ", " + eventWaitListSize + ", " + eventWaitList + ", " + retEvent + ");\n";
        wrapper += ERROR_CHECK("retCode", "OpenCL kernel enqueue error (host wrapper: \\\"" + host_func + "\\\")");
    } else {
      wrapper += "  retCode = clEnqueueNDRangeKernel(meta_queue, " + framed_kernel + ", " + std::to_string(workDim) + ", " + offset + ", " + globalSize + ", " + localSize + ", " + eventWaitListSize + ", " + eventWaitList + ", " + retEvent + ");\n";
        wrapper += ERROR_CHECK("retCode", "OpenCL kernel enqueue error (host wrapper: \\\"" + host_func + "\\\")");
    }
    wrapper += "  if (!async) {\n";
    wrapper += "    retCode = clFinish(meta_queue);\n";
    wrapper += ERROR_CHECK("retCode", "OpenCL kernel execution error (host wrapper: \\\"" + host_func + "\\\")");
    wrapper += "  }\n";
    wrapper += "  return retCode;\n";
    
    //TODO check errors and return codes/events (if sync/async)
    wrapper += "}\n";

    //Finalize the wrapper
    cache->hostFuncImpls.push_back(wrapper + "\n");
  }
}

class KernelASTConsumer : public ASTConsumer {
  public:
    KernelASTConsumer(CompilerInstance* comp, raw_ostream * out_c, raw_ostream * out_h, std::string file) : CI(comp) {
      Matcher.addMatcher(
        functionDecl(anyOf(
          functionDecl(allOf(
            hasAttr(attr::Kind::OpenCLKernel),
            isDefinition(),
            unless(hasDescendant(callExpr(callee(
              functionDecl(anyOf(
                hasName("get_global_id"),
                hasName("get_local_id"),
                hasName("get_local_linear_id"),
                hasName("get_group_id"),
                hasName("barrier")
              )).bind("nd_func")
            ))))
          )).bind("swi_kernel_def"),
          functionDecl(allOf(
            hasAttr(attr::Kind::OpenCLKernel),
            isDefinition()
          )).bind("kernel_def")
        )),
      new PrototypeHandler(out_c, out_h, file));
    }
    void HandleTranslationUnit(ASTContext &Context) override {
      //TODO Anything that needs to go here?
      Matcher.matchAST(Context);
    }

  private:
    MatchFinder Matcher;
    CompilerInstance * CI;
};

class MetaGenCLFrontendAction : public ASTFrontendAction {
  public:
    MetaGenCLFrontendAction() {}

    bool BeginInvocation(CompilerInstance &CompInst) {
      //TODO, do we need to do anything before parsing like adding headers, etc?
      return ASTFrontendAction::BeginInvocation(CompInst);
    }

    void EndSourceFileAction() {
      //TODO what do we need to do at the end of a file
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef infile) override {
      //TODO Do we need to do anything between parsing and AST Traversal?
      //TODO all this stuff needs to be moved
      //generate the cl_program
       //At the beginning of processing each new file, create the associated once-per-input-file boilerplate
       //By looking up the host code cache in the map, we force it to exist so we can populate it
      std::string file = trimFileSlashesAndType(infile.str());
      std::string outFile = ((UnifiedoutputFile.getValue() == "") ? file : UnifiedOutputFile.getValue());

	llvm::errs() << file << "\n";

      
      hostCodeCache * cache = AllHostCaches[file] = new hostCodeCache();
      //Add the core boilerplate to the hostCode cache
      //TODO add a function to metamorph for human-readable OpenCL error codes
      cache->runOnceInit += "  //TODO check for read errors if the loadprogram function doesn't already\n";
      cache->runOnceInit += "#ifdef WITH_INTELFPGA\n";
      //TODO enforce Intel name filtering to remove "kernel"
      //TODO allow them to configure the source path in an environment variable?
      cache->runOnceInit += "  progLen = metaOpenCLLoadProgramSource(\"" + file + ".aocx\", &progSrc);\n";
      cache->runOnceInit += "  __meta_gen_opencl_" + outFile + "_current_frame->" + file + "_prog = clCreateProgramWithBinary(meta_context, 1, &meta_device, &progLen, (const unsigned char **)&progSrc, NULL, &buildError);\n";
      cache->runOnceInit += "#else\n";
      //TODO allow them to configure the source path in an environment variable?
      cache->runOnceInit += "  progLen = metaOpenCLLoadProgramSource(\"" + file + ".cl\", &progSrc);\n";
      cache->runOnceInit += "  __meta_gen_opencl_" + outFile + "_current_frame->" + file + "_prog = clCreateProgramWithSource(meta_context, 1, &progSrc, &progLen, &buildError);\n";
      cache->runOnceInit += "#endif\n";
      cache->runOnceInit += ERROR_CHECK("buildError", "OpenCL program creation error");
      cache->runOnceInit += "  buildError = clBuildProgram(__meta_gen_opencl_" + outFile + "_current_frame->" + file + "_prog, 1, &meta_device, __meta_gen_opencl_" + file + "_custom_args ? __meta_gen_opencl_" + file + "_custom_args : \"\", NULL, NULL);\n";
      cache->runOnceInit += "  if (buildError != CL_SUCCESS) {\n";
      cache->runOnceInit += "    size_t logsize = 0;\n";
      cache->runOnceInit += "    clGetProgramBuildInfo(__meta_gen_opencl_" + outFile + "_current_frame->" + file + "_prog, meta_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);\n";
      cache->runOnceInit += "    char * buildLog = (char *) malloc(sizeof(char) * (logsize + 1));\n";
      cache->runOnceInit += "    clGetProgramBuildInfo(__meta_gen_opencl_" + outFile + "_current_frame->" + file + "_prog, meta_device, CL_PROGRAM_BUILD_LOG, logsize, buildLog, NULL);\n";
      cache->runOnceInit += ERROR_CHECK("buildError", "OpenCL program build error");
      cache->runOnceInit += "    fprintf(stderr, \"Build Log:\\n\%s\\n\", buildLog);\n";
      cache->runOnceInit += "    free(buildLog);\n";
      cache->runOnceInit += "  }\n";
      cache->runOnceDeinit += "    releaseError = clReleaseProgram(__meta_gen_opencl_" + outFile + "_current_frame->" + file + "_prog);\n";
      cache->runOnceDeinit += ERROR_CHECK("releaseError", "OpenCL program release error");
      if (unified_output_c != NULL) {
	cache->outfile_c = unified_output_c;
	cache->outfile_h = unified_output_h;
      } else {
        std::error_code error_c, error_h;
        //FIXME: Check error returns;
	//FIXME Filter off the .cl extension before outfile creation
        cache->outfile_c = new llvm::raw_fd_ostream(file + ".c", error_c, llvm::sys::fs::F_None);
	cache->outfile_h = new llvm::raw_fd_ostream(file + ".h", error_h, llvm::sys::fs::F_None);
        llvm::errs() << error_c.message() << error_h.message();
      }
      return llvm::make_unique<KernelASTConsumer>(&CI, cache->outfile_c, cache->outfile_h, file);
    }

  private:
    CompilerInstance *CI;
};

//For each input file in AllHostCaches, fill output files
//TODO Implement errorcode
int populateOutputFiles() {
  int errcode = 0;

  //booleans to streamline unified output checks and allow unification of code below
  bool isUnified = (UnifiedOutputFile.getValue() != "");
  bool unifiedFirstPass = true;
  std::string outFileName;

  raw_ostream * out_c, * out_h;
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    //Get the output files
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
      //headers once per output
      *out_c << "//Force MetaMorph to include the OpenCL code\n";
      *out_c << "#ifndef WITH_OPENCL\n";
      *out_c << "#define WITH_OPENCL\n";
      *out_c << "#endif\n";
      *out_c << "#include \"metamorph.h\"\n";
      *out_c << "#include \"" + outFileName + ".h\"\n";
      //Linker references for MetaMorph OpenCL variables (once per output)
      *out_c << "extern cl_context meta_context;\n";
      *out_c << "extern cl_command_queue meta_queue;\n";
      *out_c << "extern cl_device_id meta_device;\n";
    }

    //Emit user-defined types in the header file
    //FIXME ensure we only get one copy of each in unified mode
    //We use the fileCachePair output .h file as the key since we may be in unified mode but still want to get types from all input files
    for (std::pair<QualType, std::string> t : ImportedTypes[fileCachePair.second->outfile_h]) {
      *out_h << t.second;
    }

    //Generate a space to place arguments (for each input file)
    *out_c << "//TODO: Expose this with a function (with safety checks) rather than a variable\n";
    *out_c << "const char * __meta_gen_opencl_" << fileCachePair.first << "_custom_args = NULL;\n";
    *out_h << "extern const char * __meta_gen_opencl_" << fileCachePair.first << "_custom_args;\n";
    //inhibit header block for the remaining files if unified
    unifiedFirstPass = false;
  }

  //Filling out the frame has to be done as a separate loop so that the unified output correctly
  // accumulates across source files
  if (isUnified) {
    //Generate the unified module's OpenCL frame
    *out_c << "struct __meta_gen_opencl_" << outFileName << "_frame {\n";
  }
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      //Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
      //Generate the module's OpenCL frame
      *out_c << "struct __meta_gen_opencl_" << outFileName << "_frame {\n";
    }
    //Generate the program variable (one per input file)
    //TODO support one-kernel-per-program convention
    *out_c << "  cl_program " << fileCachePair.first << "_prog;\n";
    //Add the kernel variables
    for (std::string var : fileCachePair.second->cl_kernels) {
      *out_c  << var;
    }
  }

  //reset for this loop
  unifiedFirstPass = true;
  //Loop to close up the frames and *begin* initializer implementation
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      //Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      //Finish the struct and create a pointer to this module's active copy
      *out_c << "};\n";
      *out_c << "struct __meta_gen_opencl_" << outFileName << "_frame * __meta_gen_opencl_" << outFileName << "_current_frame = NULL;\n";
      *out_c << "\n";
      //Generate the MetaMorph registration function
      *out_h << "#ifdef __cplusplus\n";
      *out_h << "extern \"C\" {\n";
      *out_h << "#endif\n";
      *out_h << "a_module_record * meta_gen_opencl_" << outFileName << "_registry(a_module_record * record);\n";
      *out_c << "a_module_record * meta_gen_opencl_" << outFileName << "_registration = NULL;\n";
      *out_c << "a_module_record * meta_gen_opencl_" << outFileName << "_registry(a_module_record * record) {\n";
      *out_c << "  if (record == NULL) return meta_gen_opencl_" << outFileName << "_registration;\n";
      *out_c << "  a_module_record * old_registration = meta_gen_opencl_" << outFileName << "_registration;\n";
      *out_c << "  if (old_registration == NULL) {\n";
      *out_c << "    record->implements = module_implements_opencl;\n";
      *out_c << "    record->module_init = &meta_gen_opencl_" << outFileName << "_init;\n";
      *out_c << "    record->module_deinit = &meta_gen_opencl_" << outFileName << "_deinit;\n";
      *out_c << "    meta_gen_opencl_" << outFileName << "_registration = record;\n";
      *out_c << "  }\n";
      *out_c << "  if (old_registration != NULL && old_registration != record) return record;\n";
      *out_c << "  if (old_registration == record) meta_gen_opencl_" << outFileName << "_registration = NULL;\n";
      *out_c << "  return old_registration;\n";
      *out_c << "}\n";
      //Generate the initialization wrapper
      *out_h << "void meta_gen_opencl_" << outFileName << "_init();\n";
      *out_c << "void meta_gen_opencl_" << outFileName << "_init() {\n";
      *out_c << "  cl_int buildError, createError;\n";
      //Ensure the module is registered
      *out_c << "  if (meta_gen_opencl_" << outFileName << "_registration == NULL) {\n";
      *out_c << "    meta_register_module(&meta_gen_opencl_" << outFileName << "_registry);\n";
      *out_c << "    return;\n";
      *out_c << "  }\n";
      //Ensure a MetaMorph OpenCL state exists
      *out_c << "  if (meta_context == NULL) metaOpenCLFallBack();\n";
      //Ensure a program/kernel storage frame is initialized
      *out_c << "  if (__meta_gen_opencl_" << outFileName << "_current_frame == NULL) {\n";
      *out_c << "    __meta_gen_opencl_" << outFileName << "_current_frame = (struct __meta_gen_opencl_" << outFileName << "_frame *) malloc(sizeof(struct __meta_gen_opencl_" << outFileName << "_frame));\n";
      *out_c << "  }\n";
      *out_c << "  const char * progSrc;\n";
      *out_c << "  size_t progLen;\n";
    }
    //Add the clCreateProgram bits
    *out_c << fileCachePair.second->runOnceInit;
    //Add the clCreateKernel bits
    for ( std::string kern : fileCachePair.second->kernelInit) {
      *out_c << kern;
    }
    //inhibit registration on subsequent passes in unified mode
    unifiedFirstPass = false;
  }

  //reset for this loop
  unifiedFirstPass = true;
  //Finishing the initializers and starting the deinitializers has to be separate so the unified mode can accumulate across inputs
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      //Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      *out_c << "  meta_gen_opencl_" << outFileName << "_registration->initialized = 1;\n";
      *out_c << "}\n\n";
      //Generate the deconstruction wrapper
      *out_h << "void meta_gen_opencl_" << outFileName << "_deinit();\n";
      *out_c << "void meta_gen_opencl_" << outFileName << "_deinit() {\n";
      *out_c << "  cl_int releaseError;\n";
      //Ensure we are deregistered with MetaMorph-core
      *out_c << "  if (meta_gen_opencl_" << outFileName << "_registration != NULL) {\n";
      *out_c << "    meta_deregister_module(&meta_gen_opencl_" << outFileName << "_registry);\n";
      *out_c << "    return;\n";
      *out_c << "  }\n";
      //Esnure a program/kernel storage frame exists
      *out_c << "  if (__meta_gen_opencl_" << outFileName << "_current_frame != NULL) {\n";
    }
    //Release all the kernels
    for (std::string kern : fileCachePair.second->kernelDeinit) {
      *out_c << kern;
    }
    *out_c << fileCachePair.second->runOnceDeinit;

    //inhibit deinit generation on subsequent unified passes
    unifiedFirstPass = false;
  }

  //Reset for this loop
  unifiedFirstPass = true;
  //Free and finalize has to be a separate loop so unified mode can accumulate Releases correctly
  for (std::pair<std::string, hostCodeCache *> fileCachePair : AllHostCaches) {
    if (!isUnified) {
      //Get the output files
      out_c = fileCachePair.second->outfile_c;
      out_h = fileCachePair.second->outfile_h;
      outFileName = fileCachePair.first;
    }
    if (!isUnified || unifiedFirstPass) {
      //Release the program and frame
      *out_c << "    free(__meta_gen_opencl_" << outFileName << "_current_frame);\n";
      *out_c << "    __meta_gen_opencl_" << outFileName << "_current_frame = NULL;\n";
      *out_c << "  }\n";
      //Finish the deinit wrapper
      *out_c << "}\n\n";
    }

    //Add the kernel wrappers themselves
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
    //Inhibit frame release on subsequent unified passes
    unifiedFirstPass = false;
  }
  //Finalize separately so all wrappers are correctly accumulated
  if (isUnified) {
    out_c->flush();
    *out_h << "#ifdef __cplusplus\n";
    *out_h << "}\n";
    *out_h << "#endif\n";
    out_h->flush();
  }
  
  return errcode;
}

int main(int argc, const char ** argv) {
  int errcode = 0;
  CommonOptionsParser op(argc, argv, MetaGenCLCategory);
  //If they want unified output, generate the files
  if (UnifiedOutputFile.getValue() != "") {
    std::error_code error;
    //FIXME Check error returns
    unified_output_c = new llvm::raw_fd_ostream(UnifiedOutputFile.getValue() + ".c", error, llvm::sys::fs::F_None);
    unified_output_h = new llvm::raw_fd_ostream(UnifiedOutputFile.getValue() + ".h", error, llvm::sys::fs::F_None);
  }
  CompilationDatabase& CompDB = op.getCompilations();
  ClangTool Tool(CompDB, op.getSourcePathList());
  Tool.run(newFrontendActionFactory<MetaGenCLFrontendAction>().get());

  //After the tool runs, dump all the host code we have cached to appropriate output files.
  errcode = populateOutputFiles();
  return errcode;
}
