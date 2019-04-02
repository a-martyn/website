
from distutils.dir_util import copy_tree

from utils import *

# Globals
# -----------------------------------------
in_dir = './input'
out_dir = './output'
assets_in_pth = f'{in_dir}/assets'
assets_out_pth = f'{out_dir}/assets'
formats_with_assets = ['md', 'ipynb']
md_css = './css/markdown.css'
index_json = 'index.json'
index_template = f'{in_dir}/index.html.jinja2'
notebook_template = f'{in_dir}/notebook.html.jinja2'
# index_assets = 'input/index_assets'


# Main
# -----------------------------------------
if __name__ == "__main__":

    # Load index
    index = load_json(index_json)
    # Add urls to index hrefs
    index = add_out_pth(out_dir, index)

    # Render notebooks, add html files to index
    index = notebook2html(index, notebook_template, out_dir)

    # Render markdown, add html files to index
    index = markdown2html(index, out_dir, md_css)

    # Render index
    index2html(index, index_template, out_dir)

    # migrate assets
    migrate_assets(assets_in_pth, assets_out_pth)
    # # Compile assets, flag if duplicate
    # assets = Assets(index, formats_with_assets, index_assets)
    # assets.migrate(assets_out_pth)

