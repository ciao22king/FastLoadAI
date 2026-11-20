# FastLoadAI

CLI per scaricare e configurare modelli GGUF di Hugging Face in base alle risorse del PC, usando `llama.cpp` per l'inferenza locale.

## Installazione

Richiede Git, curl e Python 3.10 o successivo. Dopo che il repository è stato pubblicato su GitHub, installa la CLI con:

```bash
curl -fsSL https://raw.githubusercontent.com/ciao22king/FastLoadAI/main/install.sh | sh
```

Lo script crea un virtualenv isolato in `~/.local/share/fastloadai-venv`, installa il supporto Hugging Face e crea il comando `fastloadai` in `~/.local/bin`. Se `~/.local/bin` non è nel `PATH`, aggiungilo come indicato dall'installer.

Installa anche `llama.cpp` con il server e rendi disponibili nel `PATH` i binari `llama-server` e, per la quantizzazione, `llama-quantize`. FastLoadAI non compila `llama.cpp` automaticamente.

## Uso

```bash
fastloadai scan
fastloadai download bartowski/Llama-3.2-3B-Instruct-GGUF
fastloadai run
fastloadai run Llama-3.2-3B
fastloadai run --prompt "Ciao, presentati."
fastloadai plan ./modello.gguf --output config.json
fastloadai quantize ./modello.gguf ./modello-Q4_K_M.gguf --type Q4_K_M
```

`download` cerca i GGUF pubblicati nel repository e seleziona la quantizzazione di qualità più alta che rientra nella stima di memoria disponibile. Per un file specifico usa `--file`. I repository gated o privati richiedono `--token` oppure credenziali Hugging Face configurate nell'ambiente. I modelli suddivisi in shard vengono scaricati insieme.

`run` ricalcola e salva i parametri in base all'hardware rilevato, avvia `llama-server` solo su `127.0.0.1` e apre la chat FastLoadAI con risposte in streaming e cronologia. Puoi omettere il nome per usare il modello dell'ultimo `download`, oppure passare un nome presente nel repository registrato o nella cartella `models`; se ci sono più corrispondenze, specifica il percorso del GGUF. `--prompt` invia un primo messaggio. Nella chat, `/help` mostra i comandi, `/clear` azzera la cronologia e `/exit` chiude la sessione e il server. La chat non include gli strumenti di modifica file e shell dell'agente OpenCode.

Ogni configurazione registra le specifiche rilevate e i parametri runtime consigliati: thread CPU, contesto, mmap e offload GPU quando il file entra nel budget VRAM. `mmap` consente a `llama.cpp` di mappare i pesi e caricare le pagine quando servono; il sistema può comunque rallentare se la memoria fisica è insufficiente. La stima è prudenziale, non una garanzia contro l'esaurimento della memoria.

## Compatibilità

FastLoadAI genera configurazioni runtime separate e non modifica i file originali del modello. `quantize` usa `llama-quantize` su un GGUF già convertito; la conversione di checkpoint Hugging Face in GGUF non è automatizzata. I modelli MoE funzionano solo se l'architettura è supportata dalla versione di `llama.cpp` installata. Il rilevamento GPU attuale copre NVIDIA tramite `nvidia-smi`; senza GPU il piano usa CPU e mmap.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
