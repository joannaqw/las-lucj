from qiskit_ibm_runtime import QiskitRuntimeService


QiskitRuntimeService.save_account(
  token="lIW_7tU-dlMTyPK9iusA6FKYKsAg73cbujI0Qr0gxRwM", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboar
  instance="crn:v1:bluemix:public:quantum-computing:us-east:a/ed5d7d2fb3b249c6baea6864058116cb:36e451b2-5596-4f04-8689-05eced668a0e::", # Optional
  set_as_default=True, # Optional
  overwrite=True, # Optional
)


print(QiskitRuntimeService.saved_accounts())
service = QiskitRuntimeService()
print(service)
print(service.instances())
