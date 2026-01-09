import requests
import json
import os
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.hash import Hash
from solders.instruction import Instruction, AccountMeta
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.commitment_config import CommitmentLevel
from solders.rpc.requests import SendVersionedTransaction
from solders.rpc.config import RpcSendTransactionConfig
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from borsh_construct import CStruct, String, U8

# ================= НАСТРОЙКИ =================

PRIVATE_KEY = ""

TOKEN_NAME = "Legacy Coin"
TOKEN_SYMBOL = "LEGACY"
TOKEN_DESC = "Real Legacy Token V1"
IMAGE_PATH = "./example.png"

RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"

# ================= КОНСТАНТЫ =================
PUMP_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
TOKEN_PROGRAM_LEGACY = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
MPL_TOKEN_METADATA = Pubkey.from_string("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
RENT = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")

def get_pda(seeds, program_id):
    return Pubkey.find_program_address(seeds, program_id)[0]

# Оставляем только строки, байты добавим вручную
CreateArgs = CStruct(
    "name" / String,
    "symbol" / String,
    "uri" / String
)

def create_legacy_ultimate():
    # 1. Кошелек
    try:
        signer = Keypair.from_base58_string(PRIVATE_KEY)
        print(f"Кошелек: {signer.pubkey()}")
    except Exception as e:
        print(f"Ошибка ключа: {e}")
        return

    mint_keypair = Keypair()
    mint = mint_keypair.pubkey()
    print(f"Token Mint: {mint}")

    # 2. IPFS
    print("1. Загрузка IPFS...")
    try:
        with open(IMAGE_PATH, 'rb') as f:
            file_content = f.read()
        form_data = {
            'name': TOKEN_NAME,
            'symbol': TOKEN_SYMBOL,
            'description': TOKEN_DESC,
            'showName': 'true'
        }
        files = {'file': ('image.png', file_content, 'image/png')}
        ipfs_resp = requests.post("https://pump.fun/api/ipfs", data=form_data, files=files)
        if ipfs_resp.status_code != 200:
            print(f"Ошибка IPFS: {ipfs_resp.text}")
            return
        metadata_uri = ipfs_resp.json()['metadataUri']
        print(f"URI: {metadata_uri}")
    except Exception as e:
        print(f"IPFS Error: {e}")
        return

    # 3. Подготовка адресов
    bonding_curve = get_pda([b"bonding-curve", bytes(mint)], PUMP_PROGRAM)
    associated_bonding_curve = get_pda(
        [bytes(bonding_curve), bytes(TOKEN_PROGRAM_LEGACY), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM
    )
    global_state = get_pda([b"global"], PUMP_PROGRAM)
    metadata_account = get_pda(
        [b"metadata", bytes(MPL_TOKEN_METADATA), bytes(mint)],
        MPL_TOKEN_METADATA
    )
    mint_authority = get_pda([b"mint-authority"], PUMP_PROGRAM)

    # 4. Формирование инструкции
    discriminator = bytes([24, 30, 200, 40, 5, 28, 7, 119]) # GLOBAL:CREATE V1
    
    # Упаковываем строки
    args_data = CreateArgs.build({
        "name": TOKEN_NAME,
        "symbol": TOKEN_SYMBOL,
        "uri": metadata_uri
    })
    
    # Генерируем секретные 32 байта
    secret_bytes = os.urandom(32)
    
    # СКЛЕИВАЕМ ВСЁ ВМЕСТЕ: Дискриминатор + Строки + Секрет
    instruction_data = discriminator + args_data + secret_bytes

    # Аккаунты (Legacy)
    accounts = [
        AccountMeta(mint, is_signer=True, is_writable=True),
        AccountMeta(mint_authority, is_signer=False, is_writable=False),
        AccountMeta(bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(associated_bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(global_state, is_signer=False, is_writable=False),
        AccountMeta(MPL_TOKEN_METADATA, is_signer=False, is_writable=False),
        AccountMeta(metadata_account, is_signer=False, is_writable=True),
        AccountMeta(signer.pubkey(), is_signer=True, is_writable=True),
        AccountMeta(SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(TOKEN_PROGRAM_LEGACY, is_signer=False, is_writable=False), # LEGACY!
        AccountMeta(ASSOCIATED_TOKEN_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(RENT, is_signer=False, is_writable=False),
        AccountMeta(EVENT_AUTHORITY, is_signer=False, is_writable=False),
        AccountMeta(PUMP_PROGRAM, is_signer=False, is_writable=False),
    ]

    create_ix = Instruction(PUMP_PROGRAM, instruction_data, accounts)

    # Газ
    priority_fee_ix = set_compute_unit_price(100_000)
    compute_limit_ix = set_compute_unit_limit(250_000)

    # 5. Отправка
    print("2. Отправка V1 (Legacy) с секретными байтами...")
    try:
        rpc_client = requests.post(RPC_ENDPOINT, json={"jsonrpc":"2.0","id":1,"method":"getLatestBlockhash","params":[{"commitment":"finalized"}]})
        blockhash = rpc_client.json()['result']['value']['blockhash']
    except Exception as e:
        print(f"RPC Error: {e}")
        return

    msg = MessageV0.try_compile(
        payer=signer.pubkey(),
        instructions=[priority_fee_ix, compute_limit_ix, create_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.from_string(blockhash)
    )

    tx = VersionedTransaction(msg, [signer, mint_keypair])

    payload = SendVersionedTransaction(tx, RpcSendTransactionConfig(preflight_commitment=CommitmentLevel.Confirmed))

    try:
        rpc_resp = requests.post(
            url=RPC_ENDPOINT,
            headers={"Content-Type": "application/json"},
            data=payload.to_json()
        )
        result = rpc_resp.json()
        
        if 'result' in result:
            print("\n--- ПОБЕДА! ---")
            print(f"Токен: {mint}")
            print(f"Tx: https://solscan.io/tx/{result['result']}")
        else:
            print("\n--- ОШИБКА ---")
            print(result)
            
    except Exception as e:
        print(f"Ошибка сети: {e}")

if __name__ == "__main__":
    create_legacy_ultimate()
