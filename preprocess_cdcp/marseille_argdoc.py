"""Classes representing an argumentation document.

Allows unified representation.
"""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

from io import StringIO
import json
import numpy as np

# from marseille_preprocess_cdcp import merge_spans
from marseille_pdtb_fields import PDTB_FIELDS

def _transitive(links):
    """perform transitive closure of links.

    For input [(1, 2), (2, 3)] the output is [(1, 2), (2, 3), (1, 3)]
    """

    links = set(links)
    while True:
        new_links = [(src_a, trg_b)
                     for src_a, trg_a in links
                     for src_b, trg_b in links
                     if trg_a == src_b
                     and (src_a, trg_b) not in links]
        if new_links:
            links.update(new_links)
        else:
            break

    return links


def merge_prop_labels(labels):
    """After joining multiple propositions, we need to decide the new type.

    Rules:
        1. if the span is a single prop, keep the label
        2. if the span props have the same type, use that type
        3. Else, rules from Jon: policy>value>testimony>reference>fact
    """

    if len(labels) == 1:
        return labels[0]

    labels = set(labels)

    if len(labels) == 1:
        return next(iter(labels))

    if 'policy' in labels:
        return 'policy'
    elif 'value' in labels:
        return 'value'
    elif 'testimony' in labels:
        return 'testimony'
    elif 'reference' in labels:
        return 'reference'
    elif 'fact' in labels:
        return 'fact'
    else:
        raise ValueError("weird labels: {}".format(" ".join(labels)))


def merge_spans(doc, include_nonarg=True):
    """Normalization needed for CDCP data because of multi-prop spans"""

    # flatten multi-prop src spans like (3, 6) into new propositions
    # as long as they never overlap with other links. This inevitably will
    # drop some data but it's a very small number.

    # function fails if called twice because
    #    precondition: doc.links = [((i, j), k)...]
    #    postcondition: doc.links = [(i, k)...]

    new_links = []
    new_links_type = []
    new_props = {}
    new_prop_offsets = {}

    dropped = 0

    for link, link_type in zip(doc.links, doc.links_type):
        (start, end), trg = link
        if start == end:
            new_props[start] = (start, end)
            new_prop_offsets[start] = doc.prop_offsets[start]

            new_props[trg] = (trg, trg)
            new_prop_offsets[trg] = doc.prop_offsets[trg]

            new_links.append((start, trg))
            new_links_type.append(link_type)

        elif start < end:
            # multi-prop span. Check for problems:

            problems = []
            for (other_start, other_end), other_trg in doc.links:
                if start == other_start and end == other_end:
                    continue

                # another link coming out of a subset of our span
                if start <= other_start <= other_end <= end:
                    problems.append(((other_start, other_end), other_trg))

                # another link coming into a subset of our span
                if start <= other_trg <= end:
                    problems.append(((other_start, other_end), other_trg))

            if not len(problems):
                if start in new_props:
                    assert (start, end) == new_props[start]

                new_props[start] = (start, end)
                new_prop_offsets[start] = (doc.prop_offsets[start][0],
                                           doc.prop_offsets[end][1])

                new_props[trg] = (trg, trg)
                new_prop_offsets[trg] = doc.prop_offsets[trg]

                new_links.append((start, trg))
                new_links_type.append(link_type)
            else:
                # Since we drop the possibly NEW span, there is no need
                # to remove any negative links.
                dropped += 1

    if include_nonarg:
        used_props = set(k for a, b in new_props.values()
                         for k in range(a, b + 1))
        for k in range(len(doc.prop_offsets)):
            if k not in used_props:
                new_props[k] = (k, k)
                new_prop_offsets[k] = doc.prop_offsets[k]

    mapping = {key: k for k, key in enumerate(sorted(new_props))}
    doc.props = [val for _, val in sorted(new_props.items())]
    doc.prop_offsets = [val for _, val in sorted(new_prop_offsets.items())]
    doc.links = [(mapping[src], mapping[trg]) for src, trg in new_links]
    doc.links_type = new_links_type

    doc.prop_labels = [merge_prop_labels(doc.prop_labels[a:1 + b])
                       for a, b in doc.props]

    return doc


def test_merge_spans():

    load, ids = get_dataset_loader("cdcp", "train")
    n_nones = 0
    label_counts = Counter()
    for doc in load(ids):
        label_counts.update(doc.prop_labels)
        # drops 14 links in training and 8 in test split
        n_nones += sum(1 for x in doc.prop_labels if x is None)

    print(label_counts.most_common())
    print(n_nones)




def _src_to_span(src):
    if "_" in src:
        first, last = src.split("_")
        first = int(first)
        last = int(last)
    else:
        first = last = int(src)
    return first, last


def smart_join(strs):
    res = StringIO()
    res.write(strs[0])
    last = strs[0][-1]
    offsets = [0]

    for s in strs[1:]:
        offsets.append(res.tell())
        if not last.isspace() and not s[0].isspace():
            res.write(" ")
        res.write(s)
    offsets.append(res.tell())
    offsets = list(zip(offsets, offsets[1:]))
    return res.getvalue(), offsets


def process_href(span):
    begin_url = span.find("href=")
    span = span[begin_url + len("href=_"):]
    end_url = span.find('"')
    if end_url < 0:
        end_url = span.find("'")
    if end_url < 0:
        raise ValueError("url not as we expected")
    url = span[:end_url]
    span = span[end_url:]
    begin_body = span.find(">")
    end_body = span.find("<")
    body = span[1 + begin_body:end_body] + " __URL__"

    return url, body


def process_naked_link(span):
    url_begin = span.find('http')
    if url_begin < 0:
        url_begin = span.find('www')

    url_end = span.find(' ', url_begin)
    if url_end < 0:
        url_end = None
    url = span[url_begin:url_end]
    no_url = span[:url_begin] + "__URL__"
    if url_end:
        no_url += span[url_end:]
    return url, no_url


class _BaseArgumentationDoc(object):

    def __init__(self, file_root):
        if file_root is None:
            return

        self._txt_path = file_root + ".txt"
        self._nlp_path = file_root + ".txt.json"
        self._disc_path = file_root + ".txt.pipe"
        self._features_path = file_root + ".features.json"
        self._prop_features_path = file_root + ".propfeatures.json"
        self._sec_ord_features_path = file_root + ".sec_ord_features.json"

        # heavier annotations are loaded only on demand.
        self._nlp = None
        self._discourse = None
        self._features = None
        self._prop_features = None
        self._second_order_features = None
        self._compat_features = None

        # convenience properties for navigating the structure
        self._link_to_prop = None
        self._second_order = None

        with open(self._txt_path, encoding="utf8") as f:
            self.text = f.read()

    def tokens(self, key='word', lower=True):
        """Convenience function to go through document word-by-word."""
        sents = self.nlp['sentences']
        low = lambda x: x.lower() if lower else x
        return [low(w[key]) for sent in sents for w in sent['tokens']]

    @property
    def features(self):
        if self._features is not None:
            return self._features

        try:
            with open(self._features_path, encoding="utf8") as f:
                self._features = json.load(f)

            return self._features
        except FileNotFoundError:
            raise FileNotFoundError("Could not find precomputed feature file "
                                    "at {}".format(self._features_path))

    @property
    def prop_features(self):
        if self._prop_features is not None:
            return self._prop_features

        try:
            with open(self._prop_features_path, encoding="utf8") as f:
                self._prop_features = json.load(f)

            return self._prop_features
        except FileNotFoundError:
            raise FileNotFoundError("Could not find precomputed feature file "
                                    "at {}".format(self._prop_features_path))

    @property
    def second_order_features(self):
        if self._second_order_features is not None:
            return self._second_order_features

        try:
            with open(self._sec_ord_features_path, encoding='utf8') as f:
                self._second_order_features = json.load(f)
            return self._second_order_features
        except FileNotFoundError:
            raise FileNotFoundError("Could not find precomputed second order "
                                    "features at {}"
                                    "".format(self._sec_ord_features_path))

    @property
    def nlp(self):
        if self._nlp is not None:
            return self._nlp

        try:
            with open(self._nlp_path, encoding="utf8") as f:
                self._nlp = json.load(f)

            return self._nlp
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find CoreNLP output at {}".format(self._nlp_path))

    @property
    def discourse(self):
        if self._discourse is not None:
            return self._discourse

        try:
            with open(self._disc_path, encoding="utf8") as f:
                discourse = f.readlines()

            if len(discourse):
                assert len(discourse[0].split("|")) == len(PDTB_FIELDS)

            self._discourse = [dict(zip(PDTB_FIELDS, line.split("|"))) for line
                               in discourse]
            return self._discourse
        except FileNotFoundError:
            raise FileNotFoundError("Could not find discourse parser output at"
                                    " {}".format(self._disc_path))

    @property
    def compat_features(self):
        if self._compat_features is None:
            compat_f_bias = np.ones((len(self.features), 1), dtype=np.double)
            compat_f_adj = np.array([f['props_between'] == 0
                                     for f in self.features])
            compat_f_order = np.array([f['src_precedes_trg']
                                       for f in self.features])

            self._compat_features = np.column_stack([compat_f_bias,
                                                     compat_f_adj,
                                                     compat_f_order])
        return self._compat_features

    @property
    def X_compat(self):
        # TODO: deprecate me properly
        return self.compat_features

    @property
    def link_to_prop(self):
        # cache this too because it is called quite often
        if self._link_to_prop is None:
            self._link_to_prop = np.array(
                [(f['src__prop_id_'], f['trg__prop_id_'])
                 for f in self.features], dtype=np.intp)
        return self._link_to_prop

    @property
    def second_order(self):
        # precompute all possible second order links for easy indexing
        # assuming candidate edges form fully connected subgraphs
        if self._second_order is None:
            self._second_order = []
            for a, b in self.link_to_prop:
                outg = self.link_to_prop[:, 0] == b
                cs = self.link_to_prop[outg][:, 1]
                for c in cs:
                    if c != a:
                        self._second_order.append((a, b, c))
        return self._second_order

    @property
    def label(self):
        y_prop = np.array([str(f['label_']) for f in self.prop_features])
        y_link = np.array([f['label_'] for f in self.features])
        return DocLabel(y_prop, y_link)


class CdcpArgumentationDoc(_BaseArgumentationDoc):

    def __init__(self, file_root=None, merge_consecutive_spans=True):

        super(CdcpArgumentationDoc, self).__init__(file_root)

        self.doc_id = int(file_root[-5:])
        self._ann_path = file_root + ".ann.json"

        self.para_offsets = np.array([0])  # everything in same para
        # annotation is always loaded
        try:
            with open(self._ann_path, encoding="utf8") as f:
                ann = json.load(f)

            self.url = {int(key): val for key, val in ann['url'].items()}
            self.prop_labels = ann['prop_labels']
            self.prop_offsets = [(int(a), int(b))
                                 for a, b in ann['prop_offsets']]
            self.reasons = [((int(a), int(b)), int(c))
                            for (a, b), c in ann['reasons']]
            self.evidences = [((int(a), int(b)), int(c))
                              for (a, b), c in ann['evidences']]

            self.links = self.reasons + self.evidences
            self.links_type = ['reason'] * len(self.reasons) + ['evidence'] * len(self.evidences)

            self.prop_para = [0 for _ in self.prop_offsets]
        except FileNotFoundError:
            raise FileNotFoundError("Annotation json not found at {}"
                                    .format(self._ann_path))

        if merge_consecutive_spans:
            merge_spans(self)

    @staticmethod
    def from_json(doc_json):
        self = CdcpArgumentationDoc()
        self.doc_id = doc_json["commentID"]

        props = []
        self.prop_labels = []
        self.reasons = []
        self.evidences = []
        self.url = {}

        for k, prop in enumerate(doc_json['propositions']):
            assert k == prop['id']

            text = prop['text']
            if "href" in text:
                url, text = process_href(prop['text'])
                self.url[k] = url
            elif "http" in text or "www" in text:
                url, text = process_naked_link(text)
                self.url[k] = url

            props.append(text)

            self.prop_labels.append(prop['type'])

            if prop['reasons'] is not None:
                self.reasons.extend([(_src_to_span(other), k)
                                     for other in prop['reasons']])

            if prop['evidences'] is not None:
                self.evidences.extend([(_src_to_span(other), k)
                                       for other in prop['evidences']])

            self.links = self.reasons + self.evidences
        self.text, self.prop_offsets = smart_join(props)
        self.prop_para = [0 for _ in self.prop_offsets]
        return self


class UkpEssayArgumentationDoc(_BaseArgumentationDoc):

    def __init__(self, file_root=None):

        super(UkpEssayArgumentationDoc, self).__init__(file_root)

        self.doc_id = int(file_root[-3:])
        self._ann_path = file_root + ".ann"

        # get para idx

        self.para_offsets = []
        ix = 0
        while True:
            self.para_offsets.append(ix)
            try:
                ix = self.text.index("\n", ix + 1)
            except ValueError:
                break

        self.para_offsets = np.array(self.para_offsets)

        try:
            with open(self._ann_path) as f:

                props = {}
                prop_labels = {}
                prop_stances = {}
                supports = []
                attacks = []

                for line in f:
                    fields = line.strip().split("\t")

                    kind = fields[0][0]
                    num = int(fields[0][1:])
                    data = fields[1].split()

                    if kind == 'T':
                        label = data[0]
                        begin, end = int(data[1]), int(data[2])
                        props[num] = (begin, end)
                        prop_labels[num] = label

                    elif kind == 'A':
                        key, trg, val = data
                        trg = int(trg[1:])
                        assert key == 'Stance'
                        prop_stances[trg] = val

                    elif kind == 'R':
                        rel_kind = data[0]
                        src = int(data[1][6:])
                        trg = int(data[2][6:])
                        lst = supports if rel_kind == 'supports' else attacks
                        lst.append((src, trg))

                old_ix, self.prop_offsets = zip(*sorted(props.items(),
                                                        key=lambda x: x[1]))

                self.prop_para = [np.searchsorted(self.para_offsets, start) - 1
                                  for start, _ in self.prop_offsets]

                inv_idx = {k: v for v, k in enumerate(old_ix)}

                n_props = len(self.prop_offsets)

                self.prop_labels = [prop_labels[old_ix[k]]
                                    for k in range(n_props)]

                self.prop_stances = {inv_idx[k]: v
                                     for k, v in prop_stances.items()}

                self.supports = [(inv_idx[src], inv_idx[trg])
                                 for src, trg in supports]
                self.attacks = [(inv_idx[src], inv_idx[trg])
                                for src, trg in attacks]

                self.links = self.supports + self.attacks

                for src, trg in self.links:
                    assert self.prop_para[src] == self.prop_para[trg]

        except FileNotFoundError:
            raise FileNotFoundError("Annotation file not found at {}"
                                    .format(self._ann_path))


# cannot be namedtuple because of pystruct isinstance checks
class DocLabel(object):
    def __init__(self, nodes, links):
        self.nodes = nodes
        self.links = links

    def __repr__(self):
        return("DocLabel(nodes={}, links={})".format(self.nodes, self.links))


class DocStructure(object):
    def __init__(self, doc, nodes=None, links=None,
                 second_order=None):
        """Minimal convenience class for pystruct learning."""

        self.X_prop = nodes
        self.X_link = links
        self.X_compat = doc.compat_features
        self.X_sec_ord = second_order

        if nodes is not None:
            assert len(doc.prop_offsets) == nodes.shape[0]

        if links is not None:
            assert len(doc.features) == links.shape[0]

        assert links.shape[0] == self.X_compat.shape[0]

        self.prop_para = doc.prop_para
        self.link_to_prop = doc.link_to_prop
        self.second_order = doc.second_order
