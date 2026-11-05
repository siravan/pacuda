// SPDX-License-Identifier: 0BSD
//! Saved symjit applications (`.plane.symjit` of pyAmpliCol's kernel packs) read for the CUDA
//! generator: the program and the real MIR, with complex models lowered by symjit's
//! `Complexifier`, as the JIT lowers them before its code generators.
//! (From symjit-utils `src/sjb.rs`.)

use anyhow::{anyhow, Result};
use symjit::backend::{Complexifier, Mir, MirInstruction};
use symjit::{Application, CompilerType, Config, Program};

pub struct SavedModel {
    pub prog: Program,
    pub mir: Mir,
}

impl SavedModel {
    pub fn read(mut bytes: &[u8]) -> Result<SavedModel> {
        // the saved configuration replaces this one, except for its (empty) functions
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let (mut prog, reals, saved) = Application::read_sjb(&mut bytes, &config)?;
        for ins in saved.code.iter() {
            if let MirInstruction::Call { label, num_args } = &ins {
                if *num_args > 0 && saved.find_op(label).is_err() {
                    return Err(anyhow!("the kernel calls the user-defined function `{}`", label));
                }
            }
        }
        let mir = if prog.config().is_complex() {
            let mir = Complexifier::new(&reals, prog.config().clone()).complexify(&saved)?;
            if mir.consts.len() > prog.builder.consts.len() {
                prog.builder.consts = mir.consts.clone();
            }
            mir
        } else {
            saved
        };
        Ok(SavedModel { prog, mir })
    }
}
