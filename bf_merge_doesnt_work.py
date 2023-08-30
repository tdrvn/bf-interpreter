import sys
import time
import copy

def find_bracket_pairs(c):
    """
    :param c: the code for which you want to  find the bracket pairs
    :return: dict: contains for each position with a bracket its corresponding pair, None - if pairing is not correct

    Does this by keeping a stack of open parenthesis, whenever finding a closed one marking them as a pair
    and removing the last open one from the stack
    """
    pairs = {}  # dict which keeps for each position of [ or ] bracket its pair
    stack = []
    for i, char in enumerate(c):
        if char == '[':
            stack.append(i)
        elif char == ']':
            if len(stack) == 0:
                return None
            paired_index = stack[-1]
            pairs[paired_index] = i
            pairs[i] = paired_index
            stack.pop()
    if len(stack) > 0:
        # unmatched parentheses
        return None
    return pairs


def interpret(c, ip=0, mem=None, steps=None):
    if mem is None:
        mem = bytearray(30000)
    dp = 0  # data pointer
    final_line = len(c)
    if steps is not None:
        # we have a limit
        final_line = min(len(c), steps + ip)
    pairs = find_bracket_pairs(c)

    while ip < final_line:
        char = c[ip]
        ip += 1
        if char == '>':
            dp += 1
        elif char == '<':
            dp -= 1
        elif char == '+':
            mem[dp] = (mem[dp] + 1) % 256
        elif char == '-':
            # same as subtracting one (mod 256)
            mem[dp] = (mem[dp] + 255) % 256
        elif char == '.':
            print(chr(mem[dp]), end="")
        elif char == ',':
            mem[dp] = ord(sys.stdin.read(1))
        elif char == '[':
            if mem[dp] == 0:
                # ip is now at the next one
                ip = pairs[ip - 1] + 1
        elif char == ']':
            # ip is at the next one
            ip = pairs[ip - 1]
    return ip, mem


class PostponedMovement:
    """
    Class to be used for the postpone-moves optimization. Is also used for the first 3 optimizations, as they are
    just special cases of it.
    """

    def __init__(self):
        self.moves = {}
        self.fixed = {}  # for the set statements
        self.final_increase = 0

    def process_code(self, c):
        """
        :param c: Code part that has no . , [ ] as those cannot postpone

        Builds the postponed movement class for this
        """
        dp = 0
        for char in c:
            if char == '<':
                dp -= 1
            elif char == '>':
                dp += 1
            elif char == '+':
                self.moves[dp] = (self.moves.get(dp, 0) + 1) & 255
                if self.moves[dp] == 0:
                    self.moves.pop(dp)
            elif char == '-':
                self.moves[dp] = (self.moves.get(dp, 0) + 255) & 255
                if self.moves[dp] == 0:
                    self.moves.pop(dp)

        self.final_increase = dp

class OptimizedBrainfuck:
    """
    .operations represents the instructions needed. can be 4 types (currently):
        -READ

        -PRINT

        -SCAN

        -POSTPONE (compresses a <+>- sequence, allows for setting values

        -NPOSTPONE - does postpone mem[dp] times - guarantees .moves[0] is not 0 and final_increase is 0

        - NODE: While -> has a OptimizedBrainfuck son ( second param is another optimizedBrainfuck
    """

    def __init__(self, parent=None):
        self.operations = []
        self.parent = parent

def modify_nodes_scan(root):
    for i, (typescript, instruction) in enumerate(root.operations):
        # here we try changing nodes
        if typescript == 'NODE' and len(instruction.operations) == 1:
            son_type, son_instr = instruction.operations[0]
            # check that son only moves data pointer
            if son_type == 'POSTPONE':
                if len(son_instr.moves) == 0 and son_instr.final_increase != 0:
                    # we can change it to a SCAN, we also give direction
                    root.operations[i] = ('SCAN', son_instr.final_increase)

    for typescript, instruction in root.operations:
        # we try changing for sons
        if typescript == 'NODE':
            modify_nodes_scan(instruction)



def merge_operations(first, second):
    """
    :param first: First pair operation
    :param second: Second pair operation
    :return: Merges them if they are both 'POSTPONE' or if the first one is 'POSTPONE'
            and second one 'NPOSTPONE' and certain conditions are met (the value for the dp is set). Returns None otherwise
    """
    type_a, instr_a = first
    type_b, instr_b = second
    if not (type_a == type_b and type_a == 'POSTPONE'):
        # we can't merge
        return None

    # here we will try merging instr_a with instr_b
    typestring = 'POSTPONE'
    instr = PostponedMovement()
    instr.final_increase = instr_a.final_increase + instr_b.final_increase

    # this assures we keep the fixed value from the last one
    instr.fixed = copy.deepcopy(instr_a.fixed)
    instr.moves = copy.deepcopy(instr_a.moves)

    offset = instr_a.final_increase
    # we have to move all relative positions in b by offset
    for k, v in instr_b.fixed.items():
        if k + offset in instr.moves:
            # we have a set after some operation so we just set it
            instr.moves.pop(k + offset)
        instr.fixed[k + offset] = v

    for k, v in instr_b.moves.items():
        if k + offset in instr.fixed:
            # we have some operations after a set which just transform to a set
            instr.fixed[k + offset] = (instr.fixed[k + offset] + v + 256) & 255
        else:
            instr.moves[k + offset] = (instr.moves.get(k + offset, 0) + v + 256) & 255
    return typestring, instr


def modify_nodes_cancellation(root, opass=None):
    # first we try going deeper in the dfs
    new_operations = []
    for typescript, instruction in root.operations:
        # we try changing for sons
        to_add = typescript, instruction
        appended = False
        if typescript == 'NODE':
            to_add = modify_nodes_cancellation(instruction, opass)
        if len(new_operations) > 0:
            try_to_merge = merge_operations(new_operations[-1], to_add)
            if try_to_merge is not None:
                new_operations.pop()
                new_operations.append(try_to_merge)
                appended = True
        if appended is False:
            new_operations.append(to_add)

    root.operations = new_operations
    if opass == 'cancellation':
        # now we try to see if the node with it's instructions can be changed to a set statement
        # (only affects [-] type loops)
        if len(root.operations) == 1 and root.parent is not None:
            # second condition means we are in a while
            typescript, instruction = root.operations[0]
            if typescript == 'POSTPONE':
                if instruction.final_increase == 0 and len(instruction.moves) == 1 and 0 in instruction.moves and \
                        instruction.moves[0] != 0:
                    # we can change it to a set operation, we set to 0 because we assume the while is going to finish
                    set_0 = PostponedMovement()
                    set_0.fixed = instruction.fixed  # if there are some SETs in the loop we just add them  (since they are going to be set everytime)
                    set_0.fixed[0] = 0
                    return 'POSTPONE', set_0
        # we couldn't transform it
        return 'NODE', root
    else:
        # I realized after that the last optimization extends the cancellation functionality, so I've decided to
        # reuse the function but also keep the old cancellation for the first part (getting rid of [-], as I'm not sure
        # how well they would interact
        if len(root.operations) == 1 and root.parent is not None:
            typescript, instruction = root.operations[0]
            if typescript == 'POSTPONE':
                # we can change it to an n postpone if dp doesn't move and it will reach 0 in mem[dp] time:
                # verifies assumptions for NPOSTPONE
                if instruction.final_increase == 0 and (0 in instruction.moves and instruction.moves[0] != 0):
                    return 'NPOSTPONE', instruction

        return 'NODE', root


def optimize(c, opass=None):
    root_opt = OptimizedBrainfuck()
    cur_opt = root_opt  # this is the "node" I am currently at

    # this part is only first 4 optimisations, but I will need it for all (since I am using the tree for most of the
    # other optimisations

    last_char_wrong = -1  # this is the index of the last [ ' ] * character encountered, initially -1
    for i, char in enumerate(c):
        if char == '.' or char == ',' or char == '[' or char == ']':
            if i - 1 > last_char_wrong:
                postponed = PostponedMovement()
                postponed.process_code(c[last_char_wrong + 1:i])
                cur_opt.operations.append(('POSTPONE', postponed))
            last_char_wrong = i
            if char == ',':
                cur_opt.operations.append(('I/O', 'READ'))
            elif char == '.':
                cur_opt.operations.append(('I/O', 'PRINT'))
            elif char == '[':
                # make new leaf
                son = OptimizedBrainfuck(cur_opt)
                cur_opt.operations.append(('NODE', son))
                cur_opt = son
            elif char == ']':
                # go back to parent
                cur_opt = cur_opt.parent
                if cur_opt is None:
                    raise Exception("Parentheses are not right! Found a ] that matches none.")
    # case to compress last characters
    if len(c) > last_char_wrong + 1:
        postponed = PostponedMovement()
        postponed.process_code(c[last_char_wrong + 1:])
        cur_opt.operations.append(('POSTPONE', postponed))

    # I have built the tree

    if opass == 'scan' or opass is None:
        modify_nodes_scan(root_opt)

    if opass == 'cancellation' or opass == 'copy-multiply-loop-simplification' or opass is None:
        # this will just improve the POSTPONE nodes, make them also have fixed dict which allows to
        # set value to something
        modify_nodes_cancellation(root_opt, 'cancellation')

    if opass == 'copy-multiply-loop-simplification' or opass is None:
        # this adds the NPOSTPONE nodes
        modify_nodes_cancellation(root_opt)
    return root_opt  # the root


class DataPointer:
    # mutable data pointer class
    def __init__(self, val=0):
        self.val = val

    def get(self):
        return self.val

    def add(self, delta):
        self.val += delta

    def set(self, newval):
        self.val = newval

def calculate_number_of_steps(start, delta):
    # we compute the number of steps we need to start at start and reach 0 MOD256
    if delta == 255:
        return start
    if delta == 1:
        return 256 - start
    start = (start + 256) & 255
    delta = (delta + 256) & 255
    mod = 256
    while delta & 1 == 0 and start & 1 == 0 and mod > 0:
        delta //= 2
        start //= 2
        mod //= 2
    if delta % 2 == 0:
        # impossible case
        # we can never reach 0
        return None

    dif = (mod - start) & (mod - 1)
    # i want to find x st x * delta = dif (mod), where delta and mod are coprime
    x = (dif * pow(delta, -1, mod)) & (mod - 1)
    return x

def optinterpret(c: OptimizedBrainfuck, mem=None, dp=DataPointer()):
    """
    :param c: OptimizedBrainfuck code
    :param mem: buffer
    :param dp: position where we are - as mutable list of the itself
    :return: mem

    We need dp for recursion in whiles.
    Initial value for dp should be 0.
    MAKE SURE dp IS DataPointer class - TODO FIND NICER WAY TO MODIFY DP AND TO KEEP THE CHANGE
    """
    if mem is None:
        mem = bytearray(30000)
    for typestring, instruction in c.operations:
        if typestring == 'POSTPONE':
            for offset, delta in instruction.moves.items():
                mem[dp.val + offset] = (mem[dp.val + offset] + delta + 256) & 255
            for offset, setval in instruction.fixed.items():
                mem[dp.val + offset] = setval & 255

            dp.add(instruction.final_increase)
        elif typestring == 'NPOSTPONE':
            value = mem[dp.val]
            if value != 0:
                N = calculate_number_of_steps(mem[dp.val], instruction.moves[0])
                for offset, delta in instruction.moves.items():
                    mem[dp.val + offset] = (mem[dp.val + offset] + (delta + 256) * N) & 255
                for offset, setval in instruction.fixed.items():
                    mem[dp.val + offset] = setval & 255
        elif typestring == 'I/O':
            if instruction == 'READ':
                mem[dp.val] = ord(sys.stdin.read(1))
            elif instruction == 'PRINT':
                print(chr(mem[dp.val]), end="", flush=True)
        elif typestring == 'SCAN':
            if instruction == 1:
                new_dp = mem.index(0, dp.val)
                dp.set(new_dp)
            elif instruction == -1:
                new_dp = mem.rindex(0, 0, dp.val + 1)
                dp.set(new_dp)
            else:
                while mem[dp.val] != 0:
                    dp.add(instruction)
        elif typestring == 'NODE':
            # we have a son that's also a tree, execute him if while condition is true
            while mem[dp.val] > 0:
                mem = optinterpret(instruction, mem, dp)
    return mem


if __name__ == '__main__':
    #codefile = sys.argv[1]
    codefile = 'test.bf'
    with open(codefile, 'r') as fin:
        code = fin.read()

    # start_time_not_optimized = time.perf_counter()
    # mem1 = interpret(code)
    # final_time_not_optimized = time.perf_counter()
    # print(f"\nUnoptimized time: {final_time_not_optimized - start_time_not_optimized}")

    opt = optimize(code)
    start_time_optimized = time.perf_counter()
    mem2 = optinterpret(opt)
    final_time_optimized = time.perf_counter()

    print(f"\nOptimized time: {final_time_optimized - start_time_optimized}")

