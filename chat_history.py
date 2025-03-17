import json

class ChatHistory():
    def __init__(self, history_data=None):
        self.chat_history = history_data if history_data else {}

    def get_history(self, username):
        return self.chat_history[username]

    def create_user(self, username, history):
        self.chat_history[username] = history

    def remove_user(self, username):
        if username in self.chat_history.keys():
            firm = input(f"Do you want to remove data of '{username}' from the database? (Y/n or any keys to cancel): ")
            if firm.lower() == 'y':
                del self.chat_history[username]
                print('Removed')
            else:
                print('Cancelled')
                pass

    def add_history(self, username, new_history):
        if username in self.chat_history:
            self.chat_history[username].extend(new_history)

    def clear_history(self, username):
        if username in self.chat_history.keys():
            firm = input(f"Do you want to clear all the history of '{username}' from the database? (Y/n or any keys to cancel): ")
            if firm.lower() == 'y':
                del self.chat_history[username]
                print('Removed')
            else:
                print('Cancelled')
                pass

if __name__ == "__main__":
    history_data = ChatHistory()

    history_data.remove_user('P')
    history_data.remove_user('H')
    history_data.remove_user('E')

    with open('user_chat_history.json', 'w', encoding='utf-8') as f:
        json.dump(history_data.chat_history, f, ensure_ascii=False, indent=4)