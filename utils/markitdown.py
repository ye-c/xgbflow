

class MarkitDown():

    def __init__(self, content='', debug=False):
        super(MarkitDown, self).__init__()
        self.debug = debug
        self.content = content
        self.tb = ''
        self.tmp = ''

    def show(self):
        print(self.content)

    def set(self, txt):
        self.content = '%s\n\n%s' % (
            self.content, txt) if self.content else txt
        if self.debug:
            print(txt)

    def table_head(self, columns=[], align='left'):
        if not columns:
            return
        self.tb = ' | '.join(columns)
        if align == 'left':
            col_align = '|'.join([':---' for x in range(len(columns))])
        elif align == 'right':
            col_align = '|'.join(['---:' for x in range(len(columns))])
        else:
            col_align = '|'.join(['---' for x in range(len(columns))])
        self.tb = '| %s |\n| %s |' % (self.tb, col_align)
        if self.debug:
            print(self.tb)

    def tr(self, row=[]):
        if not self.tb or not isinstance(row, list):
            return
        row_str = ' | '.join(list(map(str, row)))
        self.tb += '\n| %s |' % row_str

    def table(self, columns=[], rows=[], align='left', load=True):
        if not columns:
            return
        self.tb = ' | '.join(columns)
        if align == 'left':
            col_align = ' | '.join([':---' for x in range(len(columns))])
        elif align == 'right':
            col_align = ' | '.join(['---:' for x in range(len(columns))])
        else:
            col_align = ' | '.join(['---' for x in range(len(columns))])
        self.tb = '| %s |\n| %s |' % (self.tb, col_align)
        if rows:
            for r in rows:
                # if not isinstance(r, list):
                #   continue
                r = ' | '.join(list(map(str, r)))
                self.tb += '\n| %s |' % r
        if load:
            self.load_tb()
        if self.debug:
            print(self.tb)

    def load_tb(self):
        self.content += '\n\n%s' % self.tb

    def code(self, ctype, codels):
        self.content += '\n\n```%s\n%s\n```' % (ctype, '\n'.join(codels))

    def save(self, filename='markitdown.md'):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.content)