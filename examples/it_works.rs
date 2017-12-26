extern crate wasm;

use wasm::*;
use std::collections::HashMap;

struct Calc<'a> {
    acc: &'a mut u32,
}

impl<'a> Calc<'a> {
    fn add(&mut self, operand: u32) {
        *self.acc += operand;
    }

    fn sub(&mut self, operand: u32) {
        *self.acc -= operand;
    }
}

impl<'a> Externals for Calc<'a> {
    fn invoke_index(&mut self, index: usize, args: &[Value]) -> Value {
        match index {
            0 => self.add(args[0].0),
            1 => self.sub(args[0].0),
            _ => panic!(),
        }

        Value(*self.acc)
    }

    fn signature(&self, _index: usize) -> Signature {
        Signature { arg_count: 1 }
    }
}

impl<'a> ImportResolver for Calc<'a> {
    fn resolve_fn(&self, store: &mut Store, name: &str, signature: Signature) -> FuncRef {
        let index = match name {
            "add" => 0,
            "sub" => 1,
            unknown => panic!("Calc doesnt provide fn {}", unknown),
        };
        assert_eq!(signature, self.signature(index));
        store.alloc_func_host(index, signature)
    }

    fn resolve_table(&self, store: &mut Store, _name: &str, _table_def: &TableDef) -> TableRef {
        let table_ref = store.alloc_table();
        let add_fn_ref = self.resolve_fn(store, "add", Signature { arg_count: 1 });
        store.resolve_table_mut(table_ref).set(0, add_fn_ref);
        table_ref
    }
}

fn build_module() -> Module {
    use Opcode::*;
    let main = FuncDef {
        signature: Signature { arg_count: 3 },
        body: vec![
            GetLocal(0),     // get first arg
            Call(0),         // call first import
            GetLocal(1),     // get second arg
            Call(1),         // call second import
            GetLocal(2),     //   push arg 3
            Const(Value(0)), //   push 0, index of fn inside the table
            CallIndirect(0), // call function with signature at 0 index
                             // return value from the top of the stack
        ],
    };

    let imports = vec![
        ImportDef {
            module_name: "env".into(),
            field_name: "add".into(),
            desc: ImportDesc::Func(Signature { arg_count: 1 }),
        },
        ImportDef {
            module_name: "env".into(),
            field_name: "sub".into(),
            desc: ImportDesc::Func(Signature { arg_count: 1 }),
        },
        ImportDef {
            module_name: "env".into(),
            field_name: "table".into(),
            desc: ImportDesc::Table(TableDef),
        },
    ];

    Module {
        signatures: vec![Signature { arg_count: 1 }],
        funcs: vec![main],
        imports,
        tables: vec![],
    }
}

fn main() {
    let mut acc = 0;
    let module = build_module();
    let module = validate(module);

    // Create default Store.
    let mut store = Store::default();

    {
        let mut externals = Calc { acc: &mut acc };

        let instance = instantiate(&mut store, &module, {
            let mut imports = HashMap::new();
            imports.insert("env".into(), &externals as &ImportResolver);
            imports
        });

        let args = &[Value(3), Value(2), Value(3)];
        let _ = invoke_index(&mut store, instance, 2, args, &mut externals);
    }

    println!("result = {}", acc);
}
