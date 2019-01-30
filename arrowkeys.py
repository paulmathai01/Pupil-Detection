import keyboard


min_r = 17
max_r = 61

active = 0
arr = [min_r,max_r]

def inc(x):
    x = x + 1
    
def dec(x):
    x = x - 1

def minr():
    global active
    active = 0

def maxr():
    global active
    active = 1
    
def incr():
    arr[active] = arr[active] + 1
    print(arr[active])

def decr():
    arr[active] = arr[active] - 1
    print(arr[active])

keyboard.add_hotkey('ctrl+shift+a', print, args=('triggered', 'hotkey'))
keyboard.add_hotkey('left', minr)
keyboard.add_hotkey('right', maxr)
keyboard.add_hotkey('up', incr)
keyboard.add_hotkey('down', decr)
'''
keyboard.add_hotkey('up', inc(r1_min))
keyboard.add_hotkey('down', print, r1_min)
keyboard.add_hotkey('left', print, args=('nibba', 'hotkey'))
keyboard.add_hotkey('right', print, args=('nibba', 'hotkey'))

# Press PAGE UP then PAGE DOWN to type "foobar".
keyboard.add_hotkey('page up, page down', lambda: keyboard.write('foobar'))

# Blocks until you press esc.
keyboard.wait('esc')

# Record events until 'esc' is pressed.
recorded = keyboard.record(until='esc')
# Then replay back at three times the speed.
keyboard.play(recorded, speed_factor=3)
    
# Type @@ then press space to replace with abbreviation.
keyboard.add_abbreviation('@@', 'my.long.email@example.com')

# Block forever, like `while True`.
'''
#keyboard.wait()