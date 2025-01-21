from rual.rual import RUAL

if __name__ == '__main__':
    rual_instance = RUAL()
    
    while not rual_instance.final:
        rual_instance.new_round()
