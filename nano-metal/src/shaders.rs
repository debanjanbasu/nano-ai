use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor,
    MTL4LibraryFunctionDescriptor, MTLComputePipelineState, MTLDevice, MTLLibrary,
};

use crate::Error;

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The embedded metallib plus the Metal 4 compiler that builds pipelines from it.
pub struct ShaderLibrary {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
}

impl ShaderLibrary {
    /// Load the embedded shader library onto the given device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LibraryCreation`] if the metallib data cannot be loaded.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let metallib_bytes: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
        let data = dispatch2::DispatchData::from_bytes(metallib_bytes);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|e| Error::LibraryCreation(e.to_string()))?;
        let compiler = device
            .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
            .map_err(|e| Error::LibraryCreation(e.to_string()))?;
        Ok(Self { library, compiler })
    }

    /// Name of the GPU the library was loaded on (`"Apple A15 GPU"`, `"Apple M4 Pro"`).
    #[must_use]
    pub fn device_name(&self) -> String {
        self.library.device().name().to_string()
    }

    /// Compile the compute pipeline for kernel `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PipelineCreation`] if the function is missing or fails to compile.
    pub fn pipeline(&self, name: &str) -> crate::Result<Pipeline> {
        let function = MTL4LibraryFunctionDescriptor::new();
        function.setName(Some(&NSString::from_str(name)));
        function.setLibrary(Some(&self.library));
        let descriptor = MTL4ComputePipelineDescriptor::new();
        descriptor.setComputeFunctionDescriptor(Some(&function));
        self.compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&descriptor, None)
            .map_err(|e| Error::PipelineCreation(format!("{name}: {e}")))
    }
}
