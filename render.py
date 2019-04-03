from utils import *
from config import *

# Globals
# -----------------------------------------

# Output content config
out_dir = './output'                           # path to output dir
content_dirname = 'content'                    # name of content dir in output dir
content_dir = f'{out_dir}/{content_dirname}'   # path to output content dir
assets_out_pth = f'{content_dir}/assets'       # path to output assets
assets_out_rel_pth = f'./{content_dirname}/assets' 
formats_with_assets = ['md', 'ipynb']          # input formats with linked assets

# HTML templates                 
index_template    = f'./templates/index.html.jinja2'
navbar_template   = f'./templates/navbar.html.jinja2'
notebook_template = f'./templates/notebook.html.jinja2'
markdown_template = f'./templates/markdown.html.jinja2'
comments_template = f'./templates/comments.html.jinja2'

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
    index = notebook2html(index, in_dir, templates, host, content_dir, content_dirname)

    # Render markdown, add html files to index
    index = markdown2html(index, in_dir, templates, host, content_dir, content_dirname)

    # Render index
    index2html(index, templates, out_dir, assets_out_rel_pth)

    # migrate assets
    # migrate_assets(assets_in_pth, assets_out_pth)

    # Compile assets, flag if duplicate
    assets = Assets(index, in_dir, formats_with_assets, index_assets)
    assets.migrate(assets_out_pth)

