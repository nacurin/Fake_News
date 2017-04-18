from csv import DictReader


class data():
    def __init__(self):  #all csv files should be load in same floders with .py
        bodies = "bodies.csv"
        train = "train.csv"

        self.train = self.read(train)
        articles = self.read(bodies)
        self.body = dict()

        for s in self.train:
            s['Body ID'] = int(s['Body ID'])

        for line in articles:
            self.body[int(line['Body ID'])] = line['articleBody']

    def read(self,filename):
        rows = []
        with open( filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows


class testdata():
    def __init__(self):  #all csv files should be load in same floders with .py
        bodies = "bodies.csv"
        test = "test.csv"

        self.test = self.read(test)
        articles = self.read(bodies)
        self.body = dict()

        for s in self.test:
            s['Body ID'] = int(s['Body ID'])

        for line in articles:
            self.body[int(line['Body ID'])] = line['articleBody']

    def read(self,filename):
        rows = []
        with open( filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
