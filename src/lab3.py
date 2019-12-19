import pandas as pd


class FA:
    trans_table: pd.DataFrame
    init_state: str
    accept_state: str

    def __init__(self) -> None:
        self.init_state = 'S'
        self.accept_state = 'T'
        self.trans_table = pd.DataFrame(
            index=pd.Index(data=['S', 'B', 'T']),
            columns=pd.Index(data=['a', 'b']),
            dtype=str
        )
        self.trans_table.loc['S', 'a'] = 'B'
        self.trans_table.loc['B', 'a'] = 'T'
        self.trans_table.loc['B', 'b'] = 'S'
        self.trans_table.loc['T', 'a'] = 'T'
        self.trans_table.loc['T', 'b'] = 'S'
        # print(self.trans_table)

    def judge(self, s: str):
        state = self.init_state
        for c in s:
            goto = self.trans_table.loc[state, c]
            if pd.isna(goto):
                return False
            else:
                state = goto
        return state == self.accept_state


if __name__ == '__main__':
    fa = FA()
    print(fa.judge('aababaaababaaa'))
    print(fa.judge('ababaababaaba'))
