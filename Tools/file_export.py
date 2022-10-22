# /usr/bin/env python
# -*- coding: utf - 8 - *-

"""Python utility to export files

Copyright (2018-2020), Resource Modeling Solutions Ltd. All rights reserved.
"""
import rmsp
import matplotlib.pyplot as plt
import os
import json
import shutil
from pandas.io.formats.style import Styler
import copy


class Exporter:
    """Base exporter class"""

    def __init__(self, exportpath='figures/'):
        """Initialize"""
        self.exportpath = exportpath

    def _clean_name(self, name, clean_name):
        if not clean_name:
            return name
        i = name.rfind('.')
        prefix = name[:i]
        for c in ['/', '\\', '(', ')', ']', '[', '.', ' ', ':', "%"]:
            prefix = prefix.replace(c, '_')
        prefix = prefix.lower()
        return prefix + name[i:]


class FigureExporter(Exporter):
    """A figure exporter that calls `matplotlib.savefig`"""

    def __call__(self, filename, clean_name=True, bbox_inches='tight', dpi=300,
                 **kwargs):
        """Save a figure"""
        rmsp.make_dir(self.exportpath)
        filename = self._clean_name(filename, clean_name)
        plt.savefig(os.path.join(self.exportpath, filename),
                    bbox_inches=bbox_inches, dpi=dpi, **kwargs)
        return filename


class TableExporter(Exporter):
    """A table exporter to latex using pandas"""

    def __call__(self, table, filename, clean_name=True, *args, float_format=None,
                 bold_rows=True, column_format=None, sigfigs=2, index=True,
                 multi_index=False, longtable=False, enforce_escape=False,
                 na_rep='MISS', format_table=True, **kwargs):
        """Export a table to the specified directory"""
        table = copy.deepcopy(table)
        rmsp.make_dir(self.exportpath)
        filename = self._clean_name(filename, clean_name)

        if float_format is None:
            float_format = ('{{:.{}f}}'.format(sigfigs)).format
        ncol = len(table.columns)
        if index:
            ncol += 1
        if column_format is None:
            column_format = 'c' * ncol
        if multi_index:
            column_format += 'c' * len(table.index.names)

        if isinstance(table, Styler):
            if not index:
                table.hide_index()
            if format_table:
                table = table.format(na_rep=na_rep, precision=sigfigs,
                                     escape='latex')
            table.to_latex(
                os.path.join(self.exportpath, filename), convert_css=True,
                column_format=column_format, **kwargs
            )
        else:
            table.to_latex(os.path.join(self.exportpath, filename), *args,
                           longtable=longtable, float_format=float_format,
                           index=index, bold_rows=bold_rows,
                           column_format=column_format, na_rep='-', **kwargs)
        if enforce_escape:
            self._escape_latex(os.path.join(self.exportpath, filename))

    def _escape_latex(self, filename):
        with open(filename, 'r') as f:
            content = f.read()
        content = content.replace('_', '\_')
        content = content.replace('%', '\%')
        with open(filename, 'w') as f:
            f.write(content)


class PickleExporter(Exporter):
    """A pickle export tool"""

    def __call__(self, data, filename, clean_name=True):
        """Export an object using rmsp pickle

        Args:
            data (object): input object/data to be exported
            filename (str): destindation file name for pickle

        """
        rmsp.make_dir(self.exportpath)
        filename = self._clean_name(filename, clean_name)
        rmsp.to_pickle(data, os.path.join(self.exportpath, filename))


class JsonExporter(Exporter):
    """A json export tool for dictionaries"""

    def __call__(self, dictionary, filename, clean_name=True, refresh=False):
        """Export an object using rmsp pickle

        Args:
            dictionary (dict): input dictionary to be exported
            filename (str): destindation file name for the json file

        """
        rmsp.make_dir(self.exportpath)
        filename = self._clean_name(filename, clean_name)

        file_location = os.path.join(self.exportpath, filename)

        if os.path.isfile(file_location) and not refresh:
            with open(file_location, 'r') as jfile:
                dict_current = dict(json.load(jfile))
        else:
            dict_current = {}

        dict_new = {}
        for key in list(sorted(dictionary.keys())):
            dict_new.update({key: dictionary[key]})

        dict_current.update(dict_new)

        with open(file_location, 'w', encoding='utf-8') as jf:
            json.dump(dict_current, jf, indent=4, ensure_ascii=False)


class CatPresetExporter(Exporter):
    """A json exporter for categorical color preset for Paraview"""

    def __call__(self, v_type, file_name):
        from matplotlib import colors

        paraview_preset = {}
        paraview_preset['Annotations'] = []
        paraview_preset['IndexedColors'] = []
        paraview_preset['Name'] = v_type

        cat_codes = rmsp.VariableParams[v_type]['core.alphanum_code']
        cat_caolots = rmsp.VariableParams[v_type]['plotting.cat_colors']
        for cat_name, cat_code in cat_codes.items():
            paraview_preset['Annotations'].append(f"{cat_code}")
            paraview_preset['Annotations'].append(f"{cat_name}")

            for item in colors.to_rgb(cat_caolots[cat_name]):
                paraview_preset['IndexedColors'].append(item)

        with open(file_name, 'w', encoding='utf-8') as jf:
            json.dump([paraview_preset], jf, indent=4, ensure_ascii=False)


def copytree(src, dst, symlinks=False, ignore=None):
    """Copy files and sub directories

    Args:
        src ([type]): Source directory
        dst ([type]): Destination directory
        symlinks (bool, optional): [description]. Defaults to False.
        ignore ([type], optional): [description]. Defaults to None.
    """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


save_table = TableExporter("tables/")
save_fig = FigureExporter("figures/")
