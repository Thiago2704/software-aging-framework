import subprocess
import os

# Combinação de cenarios

# para passos de 30 minutos
# Tempos de monitoramento 
splits = [
    (336, "1 semana,"),
    (48, "1 dia,"),
    (24, "12h,"),
    (2, "1h,")
]
horizontes = [
    (48, "1 dia"),
    (24, "12h"),
    (2, "1h")
]

# para passos de 10 minutos
# splits = [
#     (1008, "1 semana,"),
#     (144, "1 dia,"),
#     (72, "12h,"),
#     (6, "1h,")
# ]
# horizontes = [
#     (144, "1 dia"),
#     (72, "12h"),
#     (6, "1h")
# ]

passo_base = "Passos de 30 min"
#passo_base = "Passos de 10 min"
cenarios = []

# O laço cruza todas as possibilidades automaticamente
for split_val, split_name in splits:
    for horiz_val, horiz_name in horizontes:
        nome_pasta = f"{passo_base} Monitoramento de {split_name} Previsão de {horiz_name}"
        
        cenarios.append({
            "split": split_val,
            "horizonte": horiz_val,
            "pasta": nome_pasta
        })

# cenario especial para passos de 30 minutos
# monitoramento de 3 dias, previsão de 2 dias
cenarios.append({
    "split": 144,  
    "horizonte": 96,  
    "pasta": f"{passo_base} Monitoramento de 3 dias, Previsão de 2 dias"
})
# para passos de 10 minutos
# cenarios.append({
#     "split": 432,  
#     "horizonte": 288,  
#     "pasta": f"{passo_base} Monitoramento de 3 dias, Previsão de 2 dias"
# })


base_dir = "C:\\Área de Trabalho\\Resultados"

print(f"Iniciando bateria de {len(cenarios)} testes via linha de comando...")

for idx, cenario in enumerate(cenarios, 1):
    print(f"\n[{idx}/{len(cenarios)}] Executando cenário: Split={cenario['split']}, Horizonte={cenario['horizonte']}")

    # Cria a pasta de resultados deste cenário
    out_dir = os.path.join(base_dir, cenario['pasta'])
    os.makedirs(out_dir, exist_ok=True)
    
    # Monta o comando passando os parâmetros (CLI Flags)
    comando = (
        f"python main.py "
        f"--split_step {cenario['split']} "
        f"--horizonte {cenario['horizonte']} "
        f"--output_dir \"{out_dir}\" "
    )

    print(f" -> Rodando comando: {comando}")
    
    # Executa o main.py (o código vai travar aqui até o cenário terminar)
    subprocess.run(comando, shell=True)
    
    print(f" -> Cenário finalizado.")

print("\n Todos os experimentos foram concluídos.")