import os
import ntpath
import copy
import json
from collections import Counter
import subprocess
from distutils.file_util import copy_file
from distutils.dir_util import copy_tree
import shutil

import nbformat
from nbconvert import HTMLExporter
from jinja2 import Template

# Utils
# -----------------------------------------

def load_json(filepath: str):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def filter(key: str, value: str, index: dict):
    """Filter index by key value pair"""
    return [i for i in index if i.get(key) == value]


def add2dict(key: str, val, dict: dict):
    """Add key/value pair to dict"""
    new_dict = copy.deepcopy(dict)
    new_dict[key] = val
    return new_dict


def add_urls(index: dict):
    """Add hrefs to index for items that are url format"""
    new_index = []
    for i in index:
        if i['format'] == 'url':
            i_new = add2dict('out_pth', i['in_pth'], i)
        else:
            i_new = i
        new_index.append(i_new)
    return new_index


def data2file(data, fp: str):
    """write python data to disk at filepath `fp`"""
    with open(fp, 'w') as file:
        file.write(data)


def open_template(filepath: str):
    """Load jinja2 template"""
    with open(filepath) as file_:
        template = Template(file_.read())
    return template   


def notebook2html(index: dict, in_dir: str, templates: dict, host: str, 
                  out_dir: str, content_dirname: str):
    """
    Convert jupyter notebook to html. See relevant docs here:
    https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html#Quick-overview
    
    Possible enhancements:
    - extract images from notebook
    - render from url
    """
    new_index = []

    for item in index:
        if item.get('format') == 'ipynb':
            
            # Render notebook as html
            in_fp = f'{in_dir}/{item["in_pth"]}'
            notebook = nbformat.read(in_fp, as_version=4)
            html_exporter = HTMLExporter()
            html_exporter.template_file = 'basic'
            nb_html, resources = html_exporter.from_notebook_node(notebook)
            
            # Render navbar
            navbar = templates['navbar'].render()
            
            # Render comments section
            filename = ntpath.basename(in_fp)[:-len('.ipynb')]
            page = {'url': f'{host}/{content_dirname}/{filename}.html',
                    'identifier': filename}
            comments = templates['comments'].render(page=page)

            # Render entire page
            html = {'navbar': navbar, 'notebook': nb_html, 'comments': comments}
            body = templates['notebook'].render(html=html)
            
            # Write html to file
            out_fp = f'{out_dir}/{filename}.html'
            data2file(body, out_fp)
            
            # Add html path to index
            out_pth = f'./{content_dirname}/{filename}.html'
            item_new = add2dict('out_pth', out_pth, item)
        
        else:
            item_new = item
        new_index.append(item_new)
    return new_index


def markdown2html(index: dict, in_dir: str, templates: dict, host: str, 
                  out_dir: str, content_dirname: str):
    """
    Convert markdown to html.
    Note: wraps pandoc so depends on root pandoc install.
    """
    new_index = []

    for item in index:
        if item.get('format') == 'md':
            
            in_fp = f'{in_dir}/{item["in_pth"]}'
            # Derive output filepath
            filename = ntpath.basename(in_fp)[:-len('.md')]
            out_fp = f'{out_dir}/{filename}.html'
            
            # Use pandoc to convert markdown to html
            # with css_fp link in header
            cmd = ['pandoc',
                str(in_fp),
                '-f', 'markdown',
                '-t', 'html',
                '-o', str(out_fp)]
            print(subprocess.check_output(cmd))
            
            # Load markdown
            with open(out_fp, "r") as f:
                markdown = f.read()
            
            # Render navbar
            navbar = templates['navbar'].render() 
            
            # Render comments section
            page = {'url': f'{host}/{content_dirname}/{filename}.html',
                    'identifier': filename}
            comments = templates['comments'].render(page=page)

            # Render entire page
            html = {'navbar': navbar, 'markdown': markdown, 'comments': comments}
            body = templates['markdown'].render(html=html)

            # Write html to file
            data2file(body, out_fp)

            # Add html path to index
            out_pth = f'./{content_dirname}/{filename}.html'
            item_new = add2dict('out_pth', out_pth, item)
        else:
            item_new = item
        new_index.append(item_new)
    return new_index


def index2html(index: dict, templates: dict, out_dir: str, 
               assets_out_rel_pth: str):
    """Render index jinja template to html"""
    # Filter to only those that are set index == true
    index_visible = [i for i in index if i.get('index') == True]
    
    # Update index with derived image path
    for d in index_visible:
        d.update({'image_out_pth': f'{assets_out_rel_pth}/{d["image"]}'})

    # Render navbar
    html = {
        'navbar': templates['navbar'].render()
    }
    
    # Render index
    body = templates['index'].render(index=index_visible, html=html)
    data2file(body, f'{out_dir}/index.html')
    return index_visible


class Assets():
    """Collate assets into signle folder without duplicates"""
    def __init__(self, index: dict, in_dir: str, formats: list, 
                 index_assets_pth: str):
        self.index = index
        self.in_dir = in_dir
        self.index_assets_pth = index_assets_pth
        self.formats = formats
        self.assets = []

    def listdir(self, directory: str, ignore=['.DS_Store']):
        """
        List filepaths in directory. Ignore filenames listed in ignore
        """
        filenames = os.listdir(directory)
        filenames = [fn for fn in filenames if fn not in ignore]
        filepaths = [os.path.abspath(f'{directory}/{fn}') for fn in filenames]
        return list(zip(filenames, filepaths))

    def ls(self):
        """List all assets from input directory"""
        # List assets from homepage
        self.assets += self.listdir(self.index_assets_pth)
        # List assets for each item in index
        for item in self.index:
            # Filter items with assets
            if item['format'] in self.formats:
                # construct input assets path
                dirname = os.path.dirname(item['in_pth'])
                assets_pth = f'{self.in_dir}/{dirname}/assets'
                # list files in assets path and append to list
                if os.path.exists(assets_pth):
                    self.assets += self.listdir(assets_pth)
        return self.assets

    def validate(self):
        """Raise exception if duplicate asset filenames"""
        filenames = [i[0] for i in self.assets]
        counts = Counter(filenames)
        duplicates = {k:v for k, v in counts.items() if v > 1}
        if len(duplicates) > 0:
            raise NameError('Input assets contain duplicate filenames', 
                            duplicates)
        return

    def migrate(self, out_dir:str):
        """
        Migrate all assets to out_dir 
        provided that no duplication is found
        """
        self.ls()
        self.validate()
        # Wipe existing output images
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        # Write updated ouput images
        for filename, filepath in self.assets:
            copy_file(filepath, out_dir)


# def migrate_assets(in_dir:str, out_dir:str):
#     # Wipe existing output images
#     shutil.rmtree(out_dir)
#     os.mkdir(out_dir)
#     # copy contents
#     copy_tree(in_dir, out_dir)
