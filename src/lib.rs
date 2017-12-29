//! This is simplified version of wasm.
//!
//! Notable differences:
//! - only one value type is supported: I32,
//! - only functions and tables are modeled,
//! - functions are always return value,

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

/// Values are 32-bit integers that can be either signed or unsigned.
#[derive(Copy, Clone, Default)]
pub struct Value(pub u32);

// -----------------------------------------------------------------------------
// Wasm structure
// -----------------------------------------------------------------------------

// Indexes are zero-based. Indexes for functions and tables
// includes respective imports declared in the same module.
// The indecies of these imports precede the indicies of other definitions
// in the same index space.
type SignatureIndex = usize;
type LocalIndex = usize;
type FuncIndex = usize;

/// Signature of a function.
///
/// (i0: I32, i1: I32, .., iN: I32) -> I32
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Signature {
    pub arg_count: usize,
}

/// All instructions supported by this runtime.
#[derive(Clone)]
pub enum Opcode {
    GetLocal(LocalIndex),
    Const(Value),
    Call(FuncIndex),
    CallIndirect(SignatureIndex),
}

/// Definition of a function. Contains function's signature and the body.
#[derive(Clone)]
pub struct FuncDef {
    pub signature: Signature,
    pub body: Vec<Opcode>,
}

pub struct TableDef;

/// Import definition.
pub struct ImportDef {
    pub module_name: String,
    pub field_name: String,
    pub desc: ImportDesc,
}

/// Description of an import
pub enum ImportDesc {
    Func(Signature),
    Table(TableDef),
}

pub struct Module {
    /// List of all defined signatures.
    pub signatures: Vec<Signature>,

    /// Functions defined inside this module.
    pub funcs: Vec<FuncDef>,

    /// Tables defined inside this module.
    pub tables: Vec<TableDef>,

    /// Imports from the outside world.
    pub imports: Vec<ImportDef>,
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Module that passed validation.
pub struct ValidatedModule {
    inner: Module,
}

// Validate module. 
//
// Once module is validated, user can instantiate it.
// Complexity: O(n)
pub fn validate(module: Module) -> ValidatedModule {
    ValidatedModule {
        inner: module,
    }
}

// -----------------------------------------------------------------------------
// Runtime structures and instantiation
// -----------------------------------------------------------------------------

pub type FuncRef = Rc<FuncInstance>;
pub type TableRef = Rc<TableInstance>;
pub type ModuleRef = Rc<ModuleInstance>;

/// Function instance may represent either
/// wasm function or function defined by an embedder.
pub enum FuncInstance {
    DefinedWasm {
        func_def: FuncDef,
    },
    ImportedWasm {
        /// Module will be needed to resolve references of the code in
        /// `func`.
        module: ModuleRef,
        func_def: FuncDef,
    },
    Host { signature: Signature, index: usize },
}

impl FuncInstance {
    fn alloc_defined_wasm(func_def: FuncDef) -> FuncRef {
        Rc::new(FuncInstance::DefinedWasm { func_def })
    }

    pub fn alloc_host(index: usize, signature: Signature) -> FuncRef {
        Rc::new(FuncInstance::Host { index, signature })
    }

    fn signature(&self) -> Signature {
        match *self {
            FuncInstance::DefinedWasm { ref func_def, .. } |
            FuncInstance::ImportedWasm { ref func_def, .. } => func_def.signature,
            FuncInstance::Host { ref signature, .. } => *signature,
        }
    }

    fn func_def(&self) -> Option<&FuncDef> {
        match *self {
            FuncInstance::DefinedWasm { ref func_def, .. } | FuncInstance::ImportedWasm { ref func_def, .. } => Some(func_def),
            FuncInstance::Host { .. } => None,
        }
    }
}

/// Table instance is basically a vector of function pointers, that is 
/// primarly used to implement dynamic function dispatch.
/// 
/// A table is mutable, i.e user can change elements and grow the table.
/// Every table cell can be either function pointer or empty. If wasm code execute
/// an indirect call with an index that points to a empty table cell, then
/// trap will be raised.
pub struct TableInstance {
    elems: RefCell<Vec<Option<FuncRef>>>,
}

impl TableInstance {
    pub fn alloc() -> TableRef {
        Rc::new(TableInstance { elems: RefCell::new(vec![None; 100]) })
    }

    pub fn get(&self, index: usize) -> Option<FuncRef> {
        self.elems.borrow().get(index).unwrap().clone()
    }

    pub fn set(&self, index: usize, func: FuncRef) {
        self.elems.borrow_mut()[index] = Some(func);
    }
}

#[derive(Default)]
pub struct ModuleInstance {
    signatures: Vec<Signature>,
    funcs: Vec<FuncRef>,
    tables: Vec<TableRef>,
}

impl ModuleInstance {
    fn push_signature(&mut self, signature: Signature) {
        self.signatures.push(signature);
    }

    fn push_func(&mut self, func: FuncRef) {
        self.funcs.push(func);
    }

    fn push_table(&mut self, table: TableRef) {
        assert_eq!(
            self.tables.len(),
            0,
            "No more than 1 table is supported atm"
        );
        self.tables.push(table);
    }

    pub fn func_by_index(&self, index: FuncIndex) -> FuncRef {
        self.funcs[index].clone()
    }

    pub fn default_table(&self) -> TableRef {
        // may fail if no tables are defined/imported
        self.tables[0].clone()
    }
}

pub trait ImportResolver {
    fn resolve_fn(&self, name: &str, signature: Signature) -> FuncRef;
    fn resolve_table(&self, name: &str, table_def: &TableDef) -> TableRef;
}

// TODO: Implement ImportResolver for ModuleInstance

pub fn instantiate<'a>(module: &ValidatedModule, imports: HashMap<String, &'a ImportResolver>) -> ModuleRef {
    let module = &module.inner;

    let mut instance = ModuleInstance::default();

    for signature in &module.signatures {
        instance.push_signature(signature.clone());
    }

    // Retrieve imported instances.
    for import in &module.imports {
        let resolver = imports.get(&import.module_name).unwrap();

        let field_name = &import.field_name;
        match import.desc {
            ImportDesc::Func(signature) => {
                let func_ref = resolver.resolve_fn(field_name, signature);
                instance.push_func(func_ref);
            }

            ImportDesc::Table(ref table_def) => {
                let table_ref = resolver.resolve_table(field_name, table_def);
                instance.push_table(table_ref);
            }
        }
    }

    // Instantiate defined functions.
    for func_def in &module.funcs {
        let func_ref = FuncInstance::alloc_defined_wasm(func_def.clone());
        instance.push_func(func_ref);
    }

    // Instantiate defined tables.
    for _ in &module.tables {
        let table_ref = TableInstance::alloc();
        instance.push_table(table_ref);
    }

    Rc::new(instance)
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------

pub trait Externals {
    fn invoke_index(&mut self, index: usize, args: &[Value]) -> Value;
    fn signature(&self, index: usize) -> Signature;
}

pub fn invoke_index<E: Externals>(
    module_ref: ModuleRef,
    func_index: FuncIndex,
    args: &[Value],
    externals: &mut E,
) -> Value {
    // TODO: func_index may be out of bounds,
    let func_ref = module_ref.func_by_index(func_index);
    invoke(module_ref, func_ref, args, externals)
}

/// Invoke a function reference abstracting the actual type of the function.
fn invoke<E: Externals>(
    curr_module_ref: ModuleRef,
    invokee: FuncRef,
    args: &[Value],
    externals: &mut E,
) -> Value {
    enum InvokeKind {
        SameModule,
        DifferentModule(ModuleRef),
        Host(usize),
    }

    let invoke_kind = match *invokee {
        FuncInstance::DefinedWasm { .. } => InvokeKind::SameModule,
        FuncInstance::ImportedWasm { ref module, .. } => InvokeKind::DifferentModule(Rc::clone(module)),
        FuncInstance::Host { index, .. } => InvokeKind::Host(index),
    };

    match invoke_kind {
        InvokeKind::SameModule => evaluate(curr_module_ref, invokee, args, externals),
        InvokeKind::DifferentModule(module) => evaluate(module, invokee, args, externals),
        InvokeKind::Host(index) => externals.invoke_index(index, args)
    }
}

fn pop_args(stack: &mut Vec<Value>, signature: &Signature) -> Vec<Value> {
    let mut args = Vec::new();
    for _ in 0..signature.arg_count {
        args.insert(0, stack.pop().expect("Pop from empty stack"));
    }
    args
}

struct Frame {
    module_ref: ModuleRef,
    func_ref: FuncRef,
    locals: Vec<Value>,
    pc: usize,
}

impl Frame {
    fn new(module_ref: ModuleRef, func_ref: FuncRef, locals: Vec<Value>) -> Frame {
        assert!(func_ref.func_def().is_some(), "Frame can be created only for a function with a body");
        Frame {
            module_ref,
            func_ref,
            locals,
            pc: 0,
        }
    }
}

fn evaluate<E: Externals>(
    curr_module_ref: ModuleRef,
    func_ref: FuncRef,
    args: &[Value],
    externals: &mut E,
) -> Value {
    let mut value_stack: Vec<Value> = Vec::new();
    let mut frame_stack: Vec<Frame> = Vec::new();

    frame_stack.push(Frame::new(
        Rc::clone(&curr_module_ref),
        func_ref,
        args.to_owned(),
    ));

    loop {
        let result = {
            let frame = frame_stack.last_mut().expect(
                "a frame pushed before entry to the loop;
                frame is pushed on every a nested invoke;
                every return from an invoke pops a frame;
                if the return pops the last frame it returns from this function;
                qed"
            );
            evaluate_func(frame, &mut value_stack)
        };

        match result {
            RunResult::Return => {
                frame_stack.pop();
                if frame_stack.is_empty() {
                    // Return the value of the evaluation, since the last frame has been popped.
                    let ret = value_stack.pop().unwrap();
                    return ret;
                }
            }
            RunResult::Invoke(nested) => {
                let (module_ref, locals) = {
                    let args = pop_args(&mut value_stack, &nested.signature());
                    let module_ref = match *nested {
                        FuncInstance::DefinedWasm { .. } => Rc::clone(&curr_module_ref),
                        FuncInstance::ImportedWasm { ref module, .. } => Rc::clone(module),
                        FuncInstance::Host { index, .. } => {
                            let result = externals.invoke_index(index, &args);
                            value_stack.push(result);
                            continue;
                        }
                    };

                    (module_ref, args)
                };

                frame_stack.push(Frame::new(
                    module_ref,
                    nested,
                    locals,
                ));
            }
        }
    }
}

enum RunResult {
    Invoke(FuncRef),
    Return
}

fn evaluate_func(
    frame: &mut Frame,
    value_stack: &mut Vec<Value>,
) -> RunResult {
    let curr_module_instance = &frame.module_ref;
    let func_def = &frame.func_ref.func_def().expect("frame could contain only functions with a body; qed");
    let body = &func_def.body;

    use Opcode::*;
    loop {
        if frame.pc == body.len() {
            // Execution reached the end of the body. 
            // In that case we do implicit return.
            return RunResult::Return;
        }

        let opcode = &body[frame.pc];
        frame.pc += 1;

        match *opcode {
            Const(value) => value_stack.push(value),
            GetLocal(index) => value_stack.push(frame.locals[index]),
            Call(fn_index) => {
                let func_ref = curr_module_instance.func_by_index(fn_index);
                return RunResult::Invoke(func_ref);
            }
            CallIndirect(_signature) => {
                // Pop function index form the value stack
                let fn_index = value_stack.pop().unwrap().0 as usize;

                // Lookup function reference in the table
                let func_ref = curr_module_instance.default_table().get(fn_index).expect("Non null fn at index");

                // TODO: check expected signature against actual

                return RunResult::Invoke(func_ref);
            }
        }
    }
}
