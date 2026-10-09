"""Omit only the terminal action cache forward of the pinned single chunk.

Build from the original function's AST so scheduler, masking, noise and
decoding stay unchanged. No upstream file is edited. The caller must validate
raw/opt parity and enforce a single future latent / action chunk.
"""
import ast
import hashlib
import inspect
import textwrap
import types


def make_skip_terminal_action(policy):
    original = policy._rollout_action_chunk
    function = inspect.unwrap(original.__func__)
    assert policy.config.action_chunk_count == 2
    assert policy.config.action_chunk_steps == 8
    assert len(policy.config.future_image_indices) == 4
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    definition = tree.body[0]
    assert isinstance(definition, ast.FunctionDef)
    loops = [node for node in definition.body if isinstance(node, ast.For)]
    assert len(loops) == 1
    loop = loops[0]
    assert ast.unparse(loop.iter) == "enumerate(timesteps)"
    assert ast.unparse(loop.body[0]) == "last_step = step_id == len(timesteps) - 1"
    terminal = [node for node in loop.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "last_step"]
    assert len(terminal) == 1 and isinstance(terminal[0].body[0], ast.Continue)
    loop.body.insert(1, ast.If(test=ast.Name(id="last_step", ctx=ast.Load()),
                              body=[ast.Break()], orelse=[]))
    definition.name = "_rollout_action_chunk_skip_terminal"
    ast.fix_missing_locations(tree)
    optimized_source = ast.unparse(tree) + "\n"
    scope = dict(function.__globals__)
    exec(compile(tree, "<navsim-single-chunk-action-opt>", "exec"), scope)
    bound = types.MethodType(scope[definition.name], policy)
    return original, bound, {
        "scope": "NAVSIM single future latent/action chunk only; video terminal pass retained",
        "original_source": source, "optimized_source": optimized_source,
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "optimized_source_sha256": hashlib.sha256(optimized_source.encode()).hexdigest(),
        "transformation": "break immediately after last_step=True before terminal action forward",
    }
