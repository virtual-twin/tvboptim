"""Pretty-printing utilities for  Network Dynamics networks."""

import ast
import inspect
import textwrap
from typing import Dict, Any, List
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.core.bunch import Bunch


def format_network(network) -> str:
    """Generate pretty-printed description of a network system.

    Args:
        network: Network instance

    Returns:
        Formatted string describing the complete dynamical system
    """
    return NetworkPrinter(network).format()


def print_network(network) -> None:
    """Print a pretty-printed description of a network system.

    Args:
        network: Network instance
    """
    print(format_network(network))


class NetworkPrinter:
    """Pretty-print Network as a mathematical system."""

    def __init__(self, network):
        self.network = network
        self.coupling_descriptions = self._build_coupling_descriptions()

    def _build_coupling_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Generate descriptions for each coupling."""
        descriptions = {}
        for name, coupling in self.network.couplings.items():
            desc = CouplingDescriptor(coupling, self.network).describe()
            descriptions[name] = desc
        return descriptions

    def format(self) -> str:
        """Generate complete system description."""
        sections = [
            self._format_header(),
            self._format_graph(),
            self._format_noise(),
            self._format_couplings(),
            self._format_dynamics(),
            self._format_parameters()
        ]
        return "\n\n".join(s for s in sections if s)

    def _format_header(self) -> str:
        """Format network header."""
        lines = [
            " Network Dynamics Network System",
            "=" * 50,
            "",
            f"Dynamics: {self.network.dynamics.__class__.__name__}",
            f"  States: {', '.join(self.network.dynamics.STATE_NAMES)}",
        ]

        # Show initial state if available
        if hasattr(self.network, 'initial_state') and self.network.initial_state is not None:
            initial_str = ", ".join([
                f"{name}={val:.3g}"
                for name, val in zip(
                    self.network.dynamics.STATE_NAMES,
                    self.network.initial_state[:, 0]  # First node
                )
            ])
            lines.append(f"  Initial: {initial_str}")

        return "\n".join(lines)

    def _format_graph(self) -> str:
        """Format graph information."""
        graph = self.network.graph
        lines = [
            f"Graph: {graph.__class__.__name__}",
            f"  Nodes: {graph.n_nodes}",
        ]

        # Add density for sparse graphs
        if hasattr(graph, 'nnz'):
            density = graph.nnz / (graph.n_nodes ** 2) * 100
            lines.append(f"  Density: {density:.3f}%")

        # Add max delay for delay graphs
        if hasattr(graph, 'max_delay'):
            lines.append(f"  Max delay: {graph.max_delay} ms")

        return "\n".join(lines)

    def _format_noise(self) -> str:
        """Format noise information."""
        if not hasattr(self.network, 'noise') or self.network.noise is None:
            return ""

        noise = self.network.noise
        lines = [
            f"Noise: {noise.__class__.__name__}",
        ]

        # Show which states receive noise
        if noise.apply_to is None:
            apply_str = "all states"
        elif isinstance(noise.apply_to, str):
            apply_str = noise.apply_to
        elif isinstance(noise.apply_to, list):
            if len(noise.apply_to) > 0 and isinstance(noise.apply_to[0], str):
                # List of state names
                apply_str = f"({', '.join(noise.apply_to)})"
            else:
                # List of indices
                apply_str = f"indices {noise.apply_to}"
        else:
            apply_str = str(noise.apply_to)

        lines.append(f"  Apply to: {apply_str}")

        # Show parameters
        if hasattr(noise, 'params') and len(noise.params) > 0:
            param_items = []
            for key, value in noise.params.items():
                if isinstance(value, (int, float)):
                    param_items.append(f"{key}={value:.3g}")
                else:
                    param_items.append(f"{key}={value}")
            lines.append(f"  Params: {', '.join(param_items)}")

        return "\n".join(lines)

    def _format_couplings(self) -> str:
        """Format all couplings."""
        if not self.coupling_descriptions:
            return ""

        lines = ["Couplings", "-" * 50]

        for i, (name, desc) in enumerate(self.coupling_descriptions.items(), 1):
            lines.append(f"{i}. {name} ({desc['class_name']})")
            lines.append(f"   Type: {desc['type']}")

            # Subspace-specific information (if present)
            if 'n_regions' in desc:
                lines.append(f"   Regions: {desc['n_regions']}")
                lines.append(f"   Aggregation: {desc['aggregation']}")
                lines.append(f"   Distribution: {desc['distribution']}")

            # State information
            state_info = []
            if desc['incoming_states']:
                state_info.append(f"incoming={self._format_state_list(desc['incoming_states'])}")
            if desc['local_states']:
                state_info.append(f"local={self._format_state_list(desc['local_states'])}")
            if state_info:
                lines.append(f"   States: {', '.join(state_info)}")

            # Network form (always show)
            if desc['network_form']:
                lines.append(f"   Form: {desc['network_form']}")

            # Show pre/post if they exist
            if desc.get('pre_form'):
                lines.append(f"   pre: {desc['pre_form']}")
            if desc.get('post_form'):
                lines.append(f"   post: {desc['post_form']}")

            # Parameters (only if present)
            if desc['params']:
                param_str = ", ".join([f"{k}={v}" for k, v in desc['params'].items()])
                lines.append(f"   params: {param_str}")

            # Delay information
            if desc['type'] == 'delayed' and 'max_delay' in desc:
                lines.append(f"   Max delay: {desc['max_delay']} ms")

            lines.append("")  # Blank line between couplings

        return "\n".join(lines)

    def _format_dynamics(self) -> str:
        """Format dynamics equations."""
        formatter = DynamicsFormatter(self.network.dynamics, self.coupling_descriptions)
        return formatter.format()

    def _format_parameters(self) -> str:
        """Format parameter values."""
        if not hasattr(self.network.dynamics, 'params'):
            return ""

        params = self.network.dynamics.params
        param_items = []
        for key, value in params.items():
            if isinstance(value, (int, float)):
                param_items.append(f"{key}={value:.3g}")
            else:
                param_items.append(f"{key}={value}")

        lines = [
            "Parameters",
            "-" * 50,
            "  " + ", ".join(param_items)
        ]

        return "\n".join(lines)

    def _format_state_list(self, states) -> str:
        """Format state names."""
        if isinstance(states, str):
            return states
        elif isinstance(states, (list, tuple)):
            if len(states) == 0:
                return ""
            elif len(states) == 1:
                return states[0]
            else:
                return f"({', '.join(states)})"
        return str(states)


class CouplingDescriptor:
    """Extract mathematical form of a coupling."""

    def __init__(self, coupling, network):
        self.coupling = coupling
        self.network = network

    def describe(self) -> Dict[str, Any]:
        """Return coupling description with mathematical form.

        Returns:
            Dictionary with keys: type, incoming_states, local_states,
            params, math_form, network_form, class_name
        """
        from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling

        # Try to get description from coupling's describe() method first
        if hasattr(self.coupling, 'describe'):
            # Use coupling's describe() as the base - it has all the info we need
            result = self.coupling.describe()

            # Only add fallback fields if not already provided
            if 'class_name' not in result:
                result['class_name'] = self.coupling.__class__.__name__
            if 'type' not in result:
                result['type'] = 'delayed' if isinstance(self.coupling, DelayedCoupling) else 'instantaneous'
            if 'incoming_states' not in result:
                result['incoming_states'] = self._normalize_states(self.coupling.INCOMING_STATE_NAMES)
            if 'local_states' not in result:
                result['local_states'] = self._normalize_states(self.coupling.LOCAL_STATE_NAMES)
            if 'params' not in result:
                result['params'] = self._extract_params()
        else:
            # Fallback: build description from inspection
            result = {
                'class_name': self.coupling.__class__.__name__,
                'type': 'delayed' if isinstance(self.coupling, DelayedCoupling) else 'instantaneous',
                'incoming_states': self._normalize_states(self.coupling.INCOMING_STATE_NAMES),
                'local_states': self._normalize_states(self.coupling.LOCAL_STATE_NAMES),
                'params': self._extract_params(),
                'math_form': self._infer_from_source(),
                'network_form': self._default_network_form()
            }

        # Add delay info if applicable (only if not already provided by coupling)
        if result['type'] == 'delayed' and 'max_delay' not in result:
            if hasattr(self.network, 'max_delay'):
                result['max_delay'] = self.network.max_delay

        return result

    def _normalize_states(self, states) -> List[str]:
        """Normalize state names to list."""
        if states is None:
            return []
        elif isinstance(states, str):
            return [states]
        elif isinstance(states, (list, tuple)):
            return list(states)
        return []

    def _extract_params(self) -> Dict[str, Any]:
        """Extract coupling parameters."""
        if not hasattr(self.coupling, 'params'):
            return {}

        params = {}
        for key, value in self.coupling.params.items():
            if isinstance(value, (int, float, str)):
                params[key] = value

        return params

    def _infer_from_source(self) -> str:
        """Fallback: try to infer math form from source inspection."""
        # This is a simple fallback - the describe() method should provide better info
        return f"{self.coupling.__class__.__name__}"

    def _default_network_form(self) -> str:
        """Default network form based on coupling type."""
        from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling

        if isinstance(self.coupling, DelayedCoupling):
            return "Σⱼ w_ij * f(state_j(t - τ_ij))"
        else:
            return "Σⱼ w_ij * f(state_j(t))"


class DynamicsFormatter:
    """Parse and annotate dynamics source code."""

    def __init__(self, dynamics, coupling_descriptions):
        self.dynamics = dynamics
        self.coupling_descriptions = coupling_descriptions

    def format(self) -> str:
        """Extract and beautify dynamics equations."""
        try:
            source = inspect.getsource(self.dynamics.dynamics)

            # Remove docstring
            source = self._remove_docstring(source)

            # Collapse multi-line function signature
            source = self._collapse_function_signature(source)

            # Parse AST to find coupling usage
            coupling_usage = self._find_coupling_usage(source)

            # Annotate source with coupling descriptions
            annotated = self._annotate_source(source, coupling_usage)

            lines = [
                "Dynamics Equations",
                "-" * 50,
                annotated
            ]

            return "\n".join(lines)

        except Exception as e:
            return f"Dynamics Equations\n{'-'*50}\n# Could not parse source: {e}"

    def _remove_docstring(self, source: str) -> str:
        """Remove docstring from function source code."""
        try:
            # Dedent for parsing
            dedented = textwrap.dedent(source)
            tree = ast.parse(dedented)

            # Find the function definition
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break

            if func_def is None:
                return source

            # Check if first statement is a docstring
            if (func_def.body and
                isinstance(func_def.body[0], ast.Expr) and
                isinstance(func_def.body[0].value, (ast.Str, ast.Constant))):

                # Get the docstring node
                docstring_node = func_def.body[0]

                # Remove lines containing the docstring
                lines = source.split('\n')

                # Find docstring start and end lines
                # The first body statement starts after the function def line
                start_line = docstring_node.lineno - 1  # 0-indexed
                end_line = docstring_node.end_lineno - 1 if hasattr(docstring_node, 'end_lineno') else start_line

                # Remove docstring lines
                new_lines = lines[:start_line] + lines[end_line + 1:]

                return '\n'.join(new_lines)

            return source

        except:
            # If anything fails, return original source
            return source

    def _collapse_function_signature(self, source: str) -> str:
        """Collapse multi-line function signature into single line."""
        lines = source.split('\n')

        # Find the function definition (starts with 'def ')
        func_start = None
        func_end = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def '):
                func_start = i
                # Find where the signature ends (line ending with ':')
                for j in range(i, len(lines)):
                    if lines[j].rstrip().endswith(':'):
                        func_end = j
                        break
                break

        if func_start is None or func_end is None or func_start == func_end:
            return source

        # Extract indent from first line
        indent = lines[func_start][:len(lines[func_start]) - len(lines[func_start].lstrip())]

        # Collect signature parts, stripping each line
        sig_parts = []
        for i in range(func_start, func_end + 1):
            part = lines[i].strip()
            if part:  # Only add non-empty parts
                sig_parts.append(part)

        # Join into single line with single spaces
        collapsed = ' '.join(sig_parts)

        # Clean up spacing around parentheses and commas
        import re
        collapsed = re.sub(r'\s+', ' ', collapsed)  # Multiple spaces to single
        collapsed = re.sub(r'\s*\(\s*', '(', collapsed)  # Remove spaces around (
        collapsed = re.sub(r'\s*\)\s*', ')', collapsed)  # Remove spaces around )
        collapsed = re.sub(r'\s*,\s*', ', ', collapsed)  # Normalize comma spacing
        collapsed = re.sub(r'\s*->\s*', ' -> ', collapsed)  # Normalize arrow spacing

        # Reconstruct with collapsed signature
        new_lines = lines[:func_start] + [indent + collapsed] + lines[func_end + 1:]

        return '\n'.join(new_lines)

    def _find_coupling_usage(self, source: str) -> List[Dict[str, Any]]:
        """Find all coupling.X[Y] access patterns.

        Returns:
            List of dicts with keys: name, index, lineno, col_offset
        """
        try:
            # Dedent source to avoid indentation errors
            dedented = textwrap.dedent(source)
            tree = ast.parse(dedented)
        except:
            return []

        class CouplingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.usages = []

            def visit_Subscript(self, node):
                # Check if it's coupling.X[Y]
                if (isinstance(node.value, ast.Attribute) and
                    isinstance(node.value.value, ast.Name) and
                    node.value.value.id == 'coupling'):

                    coupling_name = node.value.attr
                    if isinstance(node.slice, (ast.Constant, ast.Index)):
                        if isinstance(node.slice, ast.Index):
                            index = node.slice.value.value if isinstance(node.slice.value, ast.Constant) else 0
                        else:
                            index = node.slice.value

                        self.usages.append({
                            'name': coupling_name,
                            'index': index,
                            'lineno': node.lineno,
                            'col_offset': node.col_offset
                        })

                self.generic_visit(node)

        visitor = CouplingVisitor()
        visitor.visit(tree)
        return visitor.usages

    def _annotate_source(self, source: str, coupling_usage: List[Dict[str, Any]]) -> str:
        """Add inline annotations for coupling usage."""
        lines = source.split('\n')

        # Group usages by line number
        usages_by_line = {}
        for usage in coupling_usage:
            lineno = usage['lineno']
            if lineno not in usages_by_line:
                usages_by_line[lineno] = []
            usages_by_line[lineno].append(usage)

        # Insert annotations (in reverse order to maintain line numbers)
        for lineno in sorted(usages_by_line.keys(), reverse=True):
            usages = usages_by_line[lineno]

            # Build annotation
            annotations = []
            for usage in usages:
                coupling_name = usage['name']
                desc = self.coupling_descriptions.get(coupling_name, {})

                if 'network_form' in desc and desc['network_form']:
                    annotations.append(f"{coupling_name}: {desc['network_form']}")
                else:
                    # Coupling referenced but not provided - defaults to 0
                    annotations.append(f"{coupling_name}: 0 (not provided)")

            if annotations:
                indent = self._get_indent(lines[lineno - 1])
                # Add annotation on the NEXT line (after the line with coupling usage)
                annotation_line = f"{indent}# ↳ {', '.join(annotations)}"
                lines.insert(lineno, annotation_line)

        return '\n'.join(lines)

    def _get_indent(self, line: str) -> str:
        """Extract indentation from a line."""
        return line[:len(line) - len(line.lstrip())]
