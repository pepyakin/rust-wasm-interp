//! Example of the nested and recursive calls.
//! 
//! There are two environments in this example: Outer and Inner.
//! The Outer environment consists of a module instance and externals object
//! that demonstrates a recurisve call to itself (i.e. makes a call to exported 
//! wasm function on the module instance) and a call to a nested environment - Inner.
//! 

extern crate wasm;

use std::rc::Rc;
use std::collections::HashMap;
use wasm::*;

mod inner {
    use wasm::*;
    use std::collections::HashMap;

    pub struct Inner<'a> {
        pub acc: &'a mut u32,
    }

    impl<'a> Inner<'a> {
        fn inc(&mut self) {
            *self.acc += 1;
        }
    }

    impl<'a> Externals for Inner<'a> {
        fn invoke_index(&mut self, index: usize, _args: &[Value]) -> Value {
            match index {
                0 => self.inc(),
                _ => panic!(),
            }
            Value(*self.acc)
        }

        fn signature(&self, _index: usize) -> Signature {
            Signature { arg_count: 0 }
        }
    }

    impl<'a> ImportResolver for Inner<'a> {
        fn resolve_fn(&self, name: &str, signature: Signature) -> FuncRef {
            let index = match name {
                "inc" => 0,
                unknown => panic!("Calc doesnt provide fn {}", unknown),
            };
            assert_eq!(signature, self.signature(index));
            FuncInstance::alloc_host(index, signature)
        }

        fn resolve_table(&self, _name: &str, _table_def: &TableDef) -> TableRef {
            unimplemented!()
        }
    }

    fn build_module() -> Module {
        use Opcode::*;
        let inner_main = FuncDef {
            signature: Signature { arg_count: 0 },
            body: vec![
                Call(0),         // call first import
            ],
        };

        let imports = vec![
            ImportDef {
                module_name: "env".into(),
                field_name: "inc".into(),
                desc: ImportDesc::Func(Signature { arg_count: 0 }),
            },
        ];

        // function index space:
        //   0: import env inc
        //   1: inner_main
        Module {
            signatures: vec![],
            funcs: vec![inner_main],
            imports,
            tables: vec![],
        }
    }

    pub fn instantiate_inner(inner: &Inner) -> ModuleRef {
        let inner_module = build_module();
        let inner_module = validate(inner_module);

        instantiate(&inner_module, {
            let mut imports = HashMap::new();
            imports.insert("env".into(), inner as &ImportResolver);
            imports
        })
    } 
}

mod outer {
    use wasm::*;

    pub struct Outer<'a> {
        pub acc: &'a mut u32,
        pub outer_instance: Option<ModuleRef>,
    }

    impl<'a> Outer<'a> {
        /// Make a call to the function in the contained module instance.
        fn mk_recursive_call(&mut self) {
            let outer_instance = self.outer_instance.as_ref().expect(
                "`outer_instance` should be set to `Some` after instantiation of the outer module instance;
                 a call to the exported function on `outer_instance` should pass this externals object;
                 this function should be invoked with outer_instance being Some;
                 qed"
            ).clone();
            
            let recursive_fn_index = 3; // see ::outer::build_module()
            let Value(v) = invoke_index(outer_instance, recursive_fn_index, &[], self);
            *self.acc += v as u32;
        }

        fn mk_nested_call(&mut self) {
            let Value(v) = {
                let mut inner = ::inner::Inner {
                    // Demonstration of the sharing of the mutable reference.
                    acc: self.acc
                };
                let inner_instance = ::inner::instantiate_inner(&inner);
                let nested_fn_index = 1; // see ::inner::build_module()
                invoke_index(inner_instance, nested_fn_index, &[], &mut inner)
            };
            *self.acc += v as u32;
        }
    }

    impl<'a> Externals for Outer<'a> {
        fn invoke_index(&mut self, index: usize, _args: &[Value]) -> Value {
            match index {
                0 => self.mk_recursive_call(),
                1 => self.mk_nested_call(),
                _ => panic!(),
            }
            Value(*self.acc)
        }

        fn signature(&self, _index: usize) -> Signature {
            Signature { arg_count: 0 }
        }
    }

    impl<'a> ImportResolver for Outer<'a> {
        fn resolve_fn(&self, name: &str, signature: Signature) -> FuncRef {
            let index = match name {
                "mk_recursive_call" => 0,
                "mk_nested_call" => 1,
                unknown => panic!("Calc doesnt provide fn {}", unknown),
            };
            assert_eq!(signature, self.signature(index));
            FuncInstance::alloc_host(index, signature)
        }

        fn resolve_table(&self, _name: &str, _table_def: &TableDef) -> TableRef {
            unimplemented!()
        }
    }

    pub fn build_module() -> Module {
        use Opcode::*;
        let main = FuncDef {
            signature: Signature { arg_count: 0 },
            body: vec![
                Call(0),         // call first import
                Call(1),         // call second import
            ],
        };

        let recursive = FuncDef {
            signature: Signature { arg_count: 0 },
            body: vec![
                Const(Value(3)),
            ],
        };

        let imports = vec![
            ImportDef {
                module_name: "env".into(),
                field_name: "mk_recursive_call".into(),
                desc: ImportDesc::Func(Signature { arg_count: 0 }),
            },
            ImportDef {
                module_name: "env".into(),
                field_name: "mk_nested_call".into(),
                desc: ImportDesc::Func(Signature { arg_count: 0 }),
            },
        ];

        // function index space:
        //   0: import env mk_recursive_call
        //   1: import env mk_nested_call
        //   2: main
        //   3: recursive
        Module {
            signatures: vec![],
            funcs: vec![main, recursive],
            imports,
            tables: vec![],
        }
    }
}

fn main() {
    let mut acc = 0;
    let outer_module = ::outer::build_module();
    let outer_module = validate(outer_module);

    {
        let mut externals = ::outer::Outer { 
            acc: &mut acc,
            outer_instance: None,
        };

        let instance = instantiate(&outer_module, {
            let mut imports = HashMap::new();
            imports.insert("env".into(), &externals as &ImportResolver);
            imports
        });
        externals.outer_instance = Some(Rc::clone(&instance));

        let args = &[];
        let _ = invoke_index(instance, 2, args, &mut externals);
    }

    println!("result = {}", acc);
}
