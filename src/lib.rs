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
    Wasm {
        /// Module will be needed to resolve references of the code in
        /// `func`.
        module: ModuleRef,
        func_def: FuncDef,
    },
    Host { signature: Signature, index: usize },
}

impl FuncInstance {
    fn alloc_wasm(module: ModuleRef, func_def: FuncDef) -> FuncRef {
        Rc::new(FuncInstance::Wasm { module, func_def })
    }

    pub fn alloc_host(index: usize, signature: Signature) -> FuncRef {
        Rc::new(FuncInstance::Host { index, signature })
    }

    fn signature(&self) -> Signature {
        match *self {
            FuncInstance::Wasm { ref func_def, .. } => func_def.signature,
            FuncInstance::Host { ref signature, .. } => *signature,
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
    signatures: RefCell<Vec<Signature>>,
    funcs: RefCell<Vec<FuncRef>>,
    tables: RefCell<Vec<TableRef>>,
}

impl ModuleInstance {
    fn push_signature(&self, signature: Signature) {
        self.signatures.borrow_mut().push(signature);
    }

    fn push_func(&self, func: FuncRef) {
        self.funcs.borrow_mut().push(func);
    }

    fn push_table(&self, table: TableRef) {
        assert_eq!(
            self.tables.borrow().len(),
            0,
            "No more than 1 table is supported atm"
        );
        self.tables.borrow_mut().push(table);
    }

    pub fn func_by_index(&self, index: FuncIndex) -> FuncRef {
        self.funcs.borrow()[index].clone()
    }

    pub fn default_table(&self) -> TableRef {
        // may fail if no tables are defined/imported
        self.tables.borrow()[0].clone()
    }
}

pub trait ImportResolver {
    fn resolve_fn(&self, name: &str, signature: Signature) -> FuncRef;
    fn resolve_table(&self, name: &str, table_def: &TableDef) -> TableRef;
}

// TODO: Implement ImportResolver for ModuleInstance

pub fn instantiate<'a>(module: &ValidatedModule, imports: HashMap<String, &'a ImportResolver>) -> ModuleRef {
    let module = &module.inner;

    let instance = Rc::new(ModuleInstance::default());

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
        let func_ref = FuncInstance::alloc_wasm(instance.clone(), func_def.clone());
        instance.push_func(func_ref);
    }

    // Instantiate defined tables.
    for _ in &module.tables {
        let table_ref = TableInstance::alloc();
        instance.push_table(table_ref);
    }

    instance
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------

pub trait Externals {
    fn invoke_index(&mut self, index: usize, args: &[Value]) -> Value;
    fn signature(&self, index: usize) -> Signature;
}

pub fn evaluate<E: Externals>(func: FuncRef, args: &[Value], externals: &mut E) -> Value {
    let (func_def, instance) = match *func {
        FuncInstance::Wasm {
            ref func_def,
            ref module,
        } => (func_def, module),
        FuncInstance::Host { index, .. } => {
            return externals.invoke_index(index, args);
        }
    };

    let locals: Vec<Value> = args.to_owned();

    let mut value_stack: Vec<Value> = Vec::new();

    let body = &func_def.body;

    use Opcode::*;
    for opcode in body {
        match *opcode {
            Const(value) => value_stack.push(value),
            GetLocal(index) => value_stack.push(locals[index]),
            Call(fn_index) => {
                let func_ref = instance.func_by_index(fn_index);
                let args = pop_args(&mut value_stack, func_ref.signature());
                let ret = evaluate(func_ref, &args, externals);
                value_stack.push(ret);
            }
            CallIndirect(_signature) => {
                // Pop function index form the value stack
                let fn_index = value_stack.pop().unwrap().0 as usize;

                // Lookup function reference in the table
                let table_ref = instance.default_table();
                let func_ref = table_ref.get(fn_index).expect("Non null fn at index");

                // TODO: check expected signature against actual

                let args = pop_args(&mut value_stack, func_ref.signature());
                let ret = evaluate(func_ref, &args, externals);

                value_stack.push(ret);
            }
        }
    }

    value_stack.pop().unwrap()
}

fn pop_args(stack: &mut Vec<Value>, signature: Signature) -> Vec<Value> {
    let mut args = Vec::new();
    for _ in 0..signature.arg_count {
        args.insert(0, stack.pop().unwrap());
    }
    args
}
