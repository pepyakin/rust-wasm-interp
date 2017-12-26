//! This is simplified version of wasm.
//!
//! Notable differences:
//! - only one value type is supported: I32,
//! - only functions and tables are modeled,
//! - functions are always return value,

use std::collections::HashMap;

/// Values are 32-bit integers that can be either signed or unsigned.
#[derive(Copy, Clone, Default, Debug)]
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
#[derive(Clone, Debug)]
pub enum Opcode {
    GetLocal(LocalIndex),
    Const(Value),
    Call(FuncIndex),
    CallIndirect(SignatureIndex),
}

/// Definition of a function. Contains function's signature and the body.
#[derive(Clone, Debug)]
pub struct FuncDef {
    pub signature: Signature,
    pub body: Vec<Opcode>,
}

#[derive(Debug)]
pub struct TableDef;

/// Import definition.
#[derive(Debug)]
pub struct ImportDef {
    pub module_name: String,
    pub field_name: String,
    pub desc: ImportDesc,
}

/// Description of an import
#[derive(Debug)]
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
    ValidatedModule { inner: module }
}

// -----------------------------------------------------------------------------
// Runtime structures and instantiation
// -----------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct FuncRef(usize);

#[derive(Copy, Clone, Debug)]
pub struct TableRef(usize);

#[derive(Copy, Clone, Debug)]
pub struct ModuleRef(usize);

/// Function instance may represent either
/// wasm function or function defined by an embedder.
#[derive(Debug)]
pub enum FuncInstance {
    ImportedWasm {
        module: ModuleRef,
        func_def: Option<FuncDef>,
    },
    DefinedWasm { 
        func_def: Option<FuncDef> 
    },
    Host { signature: Signature, index: usize },
}

impl FuncInstance {
    fn signature(&self) -> Signature {
        match *self {
            FuncInstance::DefinedWasm { ref func_def, .. } |
            FuncInstance::ImportedWasm { ref func_def, .. } => func_def.as_ref().expect("func definition is taken").signature.clone(),
            FuncInstance::Host { ref signature, .. } => *signature,
        }
    }

    fn func_def_mut(&mut self) -> &mut Option<FuncDef> {
        match *self {
            FuncInstance::DefinedWasm { ref mut func_def, .. } | 
            FuncInstance::ImportedWasm { ref mut func_def, .. } => func_def,
            _ => panic!(),
        }
    }

    fn take_func_def(&mut self) -> FuncDef {
        self.func_def_mut().take().expect("func definition already taken")
    }

    fn put_back_func_def(&mut self, func_def: FuncDef) {
        assert!(self.func_def_mut().is_none());
        *self.func_def_mut() = Some(func_def);
    }
}

/// Table instance is basically a vector of function pointers, that is
/// primarly used to implement dynamic function dispatch.
///
/// A table is mutable, i.e user can change elements and grow the table.
/// Every table cell can be either function pointer or empty. If wasm code execute
/// an indirect call with an index that points to a empty table cell, then
/// trap will be raised.
#[derive(Debug)]
pub struct TableInstance {
    elems: Vec<Option<FuncRef>>,
}

impl TableInstance {
    pub fn get(&self, index: usize) -> Option<FuncRef> {
        self.elems[index].clone()
    }

    pub fn set(&mut self, index: usize, func: FuncRef) {
        self.elems[index] = Some(func);
    }
}

#[derive(Default, Debug)]
pub struct Store {
    funcs: Vec<FuncInstance>,
    tables: Vec<TableInstance>,
    modules: Vec<Option<ModuleInstance>>,
}

impl Store {
    fn alloc_func_defined(&mut self, func_def: FuncDef) -> FuncRef {
        let id = FuncRef(self.funcs.len());
        self.funcs.push(FuncInstance::DefinedWasm { func_def: Some(func_def) });
        id
    }

    pub fn alloc_func_host(&mut self, index: usize, signature: Signature) -> FuncRef {
        let id = FuncRef(self.funcs.len());
        self.funcs.push(FuncInstance::Host { index, signature });
        id
    }

    pub fn alloc_table(&mut self) -> TableRef {
        let id = TableRef(self.tables.len());
        self.tables.push(TableInstance {
            elems: vec![None; 100],
        });
        id
    }

    fn alloc_module(&mut self, instance: ModuleInstance) -> ModuleRef {
        let id = ModuleRef(self.modules.len());
        self.modules.push(Some(instance));
        id
    }

    pub fn resolve_module(&self, module: ModuleRef) -> &ModuleInstance {
        &self.modules[module.0].as_ref().expect("Module is loaned")
    }

    pub fn resolve_func(&self, func: FuncRef) -> &FuncInstance {
        &self.funcs[func.0]
    }

    pub fn resolve_table(&self, table: TableRef) -> &TableInstance {
        &self.tables[table.0]
    }

    pub fn resolve_table_mut(&mut self, table: TableRef) -> &mut TableInstance {
        &mut self.tables[table.0]
    }

    fn take_module(&mut self, module: ModuleRef) -> ModuleInstance {
        self.modules[module.0].take().unwrap()
    }

    fn put_back_module(&mut self, module: ModuleRef, instance: ModuleInstance) {
        self.modules[module.0] = Some(instance);
    }

    fn take_func_def(&mut self, func: FuncRef) -> FuncDef {
        self.funcs[func.0].take_func_def()
    }

    fn put_back_func_def(&mut self, func: FuncRef, func_def: FuncDef) {
        self.funcs[func.0].put_back_func_def(func_def)
    }
}

#[derive(Default, Debug)]
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
    fn resolve_fn(&self, store: &mut Store, name: &str, signature: Signature) -> FuncRef;
    fn resolve_table(&self, store: &mut Store, name: &str, table_def: &TableDef) -> TableRef;
}

// TODO: Implement ImportResolver for ModuleInstance

pub fn instantiate<'a>(
    store: &mut Store,
    module: &ValidatedModule,
    imports: HashMap<String, &'a ImportResolver>,
) -> ModuleRef {
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
                let func_ref = resolver.resolve_fn(store, field_name, signature);
                instance.push_func(func_ref);
            }

            ImportDesc::Table(ref table_def) => {
                let table_ref = resolver.resolve_table(store, field_name, table_def);
                instance.push_table(table_ref);
            }
        }
    }

    // Instantiate defined functions.
    for func_def in &module.funcs {
        let func_ref = store.alloc_func_defined(func_def.clone());
        instance.push_func(func_ref);
    }

    // Instantiate defined tables.
    for _ in &module.tables {
        let table_ref = store.alloc_table();
        instance.push_table(table_ref);
    }

    store.alloc_module(instance)
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------

pub trait Externals {
    fn invoke_index(&mut self, index: usize, args: &[Value]) -> Value;
    fn signature(&self, index: usize) -> Signature;
}

pub fn invoke_index<E: Externals>(
    store: &mut Store,
    module_ref: ModuleRef,
    func_index: FuncIndex,
    args: &[Value],
    externals: &mut E,
) -> Value {
    // TODO: func_index may be out of bounds,
    let func_ref = store.resolve_module(module_ref).func_by_index(func_index);
    invoke(store, module_ref, func_ref, args, externals)
}

/// Invoke a function reference abstracting the actual type of the function.
fn invoke<E: Externals>(
    store: &mut Store,
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

    let invoke_kind = match *store.resolve_func(invokee) {
        FuncInstance::DefinedWasm { .. } => InvokeKind::SameModule,
        FuncInstance::ImportedWasm { ref module, .. } => InvokeKind::DifferentModule(*module),
        FuncInstance::Host { index, .. } => InvokeKind::Host(index),
    };

    match invoke_kind {
        InvokeKind::SameModule => evaluate(store, curr_module_ref, invokee, args, externals),
        InvokeKind::DifferentModule(module) => evaluate(store, module, invokee, args, externals),
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

fn evaluate<E: Externals>(
    store: &mut Store,
    curr_module_ref: ModuleRef,
    func_ref: FuncRef,
    args: &[Value],
    externals: &mut E,
) -> Value {
    let mut value_stack: Vec<Value> = Vec::new();
    let mut frame_stack: Vec<Frame> = Vec::new();

    frame_stack.push(Frame {
        module_ref: curr_module_ref,
        func_ref: func_ref,
        locals: args.to_owned(),
        pc: 0,
    });

    loop {
        let mut frame = frame_stack.pop().unwrap();

        let module_instance = store.take_module(frame.module_ref);
        let func_def = store.take_func_def(frame.func_ref);

        let result = evaluate_func(store, &module_instance, &func_def, &mut frame, &mut value_stack);

        store.put_back_module(frame.module_ref, module_instance);
        store.put_back_func_def(frame.func_ref, func_def);

        frame_stack.push(frame);

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
                    let func_instance = store.resolve_func(nested);
                    let args = pop_args(&mut value_stack, &func_instance.signature());
                    let module_ref = match *func_instance {
                        FuncInstance::DefinedWasm { .. } => curr_module_ref,
                        FuncInstance::ImportedWasm { ref module, .. } => *module,
                        FuncInstance::Host { index, .. } => {
                            let result = externals.invoke_index(index, &args);
                            value_stack.push(result);
                            continue;
                        }
                    };

                    (module_ref, args)
                };

                frame_stack.push(Frame {
                    module_ref,
                    func_ref: nested,
                    locals,
                    pc: 0,
                });
            }
        }
    }
}

enum RunResult {
    Invoke(FuncRef),
    Return
}

fn evaluate_func(
    store: &mut Store,
    curr_module_instance: &ModuleInstance,
    func_def: &FuncDef,
    frame: &mut Frame,
    value_stack: &mut Vec<Value>,
) -> RunResult {
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
                let table_ref = curr_module_instance.default_table();
                let func_ref = store.resolve_table(table_ref).get(fn_index).expect("Non null fn at index");

                // TODO: check expected signature against actual

                return RunResult::Invoke(func_ref);
            }
        }
    }
}

#[test]
fn store_is_send_and_sync() {
    fn assert_send<T: Send>() { }
    fn assert_sync<T: Sync>() { }

    assert_sync::<Store>();
    assert_send::<Store>();
}
