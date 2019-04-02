
from distutils.dir_util import copy_tree

from utils import *

# Globals
# -----------------------------------------
in_dir = './input/content'
out_dir = './output'
content_dirname = 'content'
content_dir = f'{out_dir}/{content_dirname}'

assets_in_pth = f'{in_dir}/assets'
assets_out_pth = f'{content_dir}/assets'
formats_with_assets = ['md', 'ipynb']
md_css = './css/markdown.css'
index_json = 'index.json'
index_template    = f'./input/templates/index.html.jinja2'
navbar_template   = f'./input/templates/navbar.html.jinja2'
notebook_template = f'./input/templates/notebook.html.jinja2'
markdown_template = f'./input/templates/markdown.html.jinja2'
comments_template = f'./input/templates/comments.html.jinja2'
host = 'alanmartyn.com'

# Load html templates
# -----------------------------------------
templates = {
    'index': open_template(index_template),
    'navbar': open_template(navbar_template),
    'notebook': open_template(notebook_template),
    'markdown': open_template(markdown_template),
    'comments': open_template(comments_template)
}

# Main
# -----------------------------------------
if __name__ == "__main__":

    # Load index
    index = load_json(index_json)
    # Add urls to index hrefs
    index = add_urls(index)

    # Render notebooks, add html files to index
    index = notebook2html(index, templates, host, content_dir, content_dirname)

    # Render markdown, add html files to index
    index = markdown2html(index, templates, host, content_dir, content_dirname)

    # Render index
    index2html(index, templates, out_dir)

    # migrate assets
    migrate_assets(assets_in_pth, assets_out_pth)
    # # Compile assets, flag if duplicate
    # assets = Assets(index, formats_with_assets, index_assets)
    # assets.migrate(assets_out_pth)

