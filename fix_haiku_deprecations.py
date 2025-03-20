#!/usr/bin/env python3
"""
Script to fix JAX deprecation warnings in Haiku.
This script patches the haiku/_src/jaxpr_info.py file to replace
jax.core with jax.extend.core for the deprecated imports.
"""

import os
import sys
import re
from pathlib import Path

def find_haiku_jaxpr_info():
    """Find the path to the haiku jaxpr_info.py file."""
    # Try to find the site-packages directory
    try:
        import haiku
        haiku_path = Path(haiku.__file__).parent
        jaxpr_info_path = haiku_path / "_src" / "jaxpr_info.py"
        if jaxpr_info_path.exists():
            return str(jaxpr_info_path)
    except ImportError:
        pass
    
    # Fall back to searching in common site-packages locations
    import site
    for site_dir in site.getsitepackages():
        jaxpr_info_path = Path(site_dir) / "haiku" / "_src" / "jaxpr_info.py"
        if jaxpr_info_path.exists():
            return str(jaxpr_info_path)
    
    return None

def backup_file(filepath):
    """Create a backup of the file."""
    backup_path = f"{filepath}.bak"
    with open(filepath, 'r') as src:
        with open(backup_path, 'w') as dst:
            dst.write(src.read())
    return backup_path

def patch_jaxpr_info(filepath):
    """Patch the jaxpr_info.py file to fix JAX deprecation warnings."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace jax.core with jax.extend.core
    pattern = r'jax\.core\.(JaxprEqn|Var|Jaxpr)'
    replacement = r'jax.extend.core.\1'
    patched_content = re.sub(pattern, replacement, content)
    
    # Update imports if needed
    if 'from jax import core' in content:
        patched_content = patched_content.replace(
            'from jax import core', 
            'from jax.extend import core'
        )
    
    if 'import jax.core' in content:
        patched_content = patched_content.replace(
            'import jax.core', 
            'import jax.extend.core'
        )
    
    # Write the patched file
    with open(filepath, 'w') as f:
        f.write(patched_content)
    
    return True

def main():
    """Main function to patch Haiku files."""
    jaxpr_info_path = find_haiku_jaxpr_info()
    
    if not jaxpr_info_path:
        print("Error: Could not find haiku/_src/jaxpr_info.py")
        sys.exit(1)
    
    print(f"Found jaxpr_info.py at: {jaxpr_info_path}")
    
    # Backup the file
    backup_path = backup_file(jaxpr_info_path)
    print(f"Created backup at: {backup_path}")
    
    # Patch the file
    if patch_jaxpr_info(jaxpr_info_path):
        print(f"Successfully patched {jaxpr_info_path} to use jax.extend.core")
        print("This should resolve the JAX deprecation warnings")
    else:
        print(f"Failed to patch {jaxpr_info_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 